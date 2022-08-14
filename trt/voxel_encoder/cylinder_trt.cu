#include "common/hash32.h"
#include "common/launch.cuh"
#include "common/macros.h"
#include "common/refnd.h"
#include <NvInfer.h>
#include <cstring>
#include <cub/cub.cuh>
#include <vector>

#define NOEXCEPT noexcept

namespace voxel_encoder {
using utils::GPU;
using utils::io::SerializeStream;
using utils::launch::CUDA_NUM_THREADS;
using utils::launch::getBlocks;
using utils::launch::KernelLoopX;
using utils::nd::fromTensorRT;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;
using utils::nd::Size;
using utils::nd::Vec;
using namespace nvinfer1;

namespace kernel {
using HashTable = hash::LinearHashTable<int32_t, int32_t>;
template <class T, class Hash>
__global__ void cylinderize(Ref1D<int32_t> hashSlot,  // [numPtsIn]
                            Hash hash,
                            Ref2D<T> batchAugPointFeats,  // [numPtsIn, 9]
                            const Ref2D<const T> batchPointFeats,  // [numPtsIn, 4]
                            const Ref1D<const int32_t> batchIndices,  // [numPtsIn]
                            const Ref1D<const T> cylinderConfig,  // [6]
                            const Size<4> inSpatialShape) {
  auto numPtsIn = batchPointFeats.size(0);
  for (size_t ix : KernelLoopX(numPtsIn)) {
    const T* ptsFeats = &batchPointFeats(ix, 0);
    T* ptsAugFeats = &batchAugPointFeats(ix, 0);
    const auto numFeats = batchPointFeats.size(1);
    bool notNan = true;
    for (int i = 0; i < numFeats; i++) { notNan = notNan && isfinite(ptsFeats[i]); }
    if (notNan) {
      Vec<4, int32_t> ptsCoors{};
      ptsCoors[0] = batchIndices[ix];
      T x = ptsFeats[0];
      T y = ptsFeats[1];
      T z = ptsFeats[2];
      T rho = sqrtf(x * x + y * y);
      T phi = atan2f(y, x);
      if (ptsCoors[0] >= 0) {
        T zMin = cylinderConfig[0];
        T phiMin = cylinderConfig[1];
        T rhoMin = cylinderConfig[2];
        auto rhoDim = inSpatialShape[1];
        auto phiDim = inSpatialShape[2];
        auto zDim = inSpatialShape[3];
        T zSize = (cylinderConfig[3] - cylinderConfig[0]) / zDim;
        T phiSize = (cylinderConfig[4] - cylinderConfig[1]) / phiDim;
        T rhoSize = (cylinderConfig[5] - cylinderConfig[2]) / rhoDim;
        auto zCoors = static_cast<int32_t>((z - zMin) / zSize);
        auto phiCoors = static_cast<int32_t>((phi - phiMin) / phiSize);
        auto rhoCoors = static_cast<int32_t>((rho - rhoMin) / rhoSize);
        zCoors = zCoors < 0 ? 0 : zCoors;
        phiCoors = phiCoors < 0 ? 0 : phiCoors;
        rhoCoors = rhoCoors < 0 ? 0 : rhoCoors;
        ptsCoors[3] = zCoors >= zDim ? (zDim - 1) : zCoors;
        ptsCoors[2] = phiCoors >= phiDim ? (phiDim - 1) : phiCoors;
        ptsCoors[1] = rhoCoors >= rhoDim ? (rhoDim - 1) : rhoCoors;
        hashSlot[ix] = hash.insert(inSpatialShape.template offset(ptsCoors), 1);

        ptsAugFeats[0] = z;
        ptsAugFeats[1] = phi;
        ptsAugFeats[2] = rho;
        ptsAugFeats[3] = z - ((zCoors + T(0.5)) * zSize + zMin);
        ptsAugFeats[4] = phi - ((phiCoors + T(0.5)) * phiSize + phiMin);
        ptsAugFeats[5] = rho - ((rhoCoors + T(0.5)) * rhoSize + rhoMin);
        ptsAugFeats[6] = x;
        ptsAugFeats[7] = y;
        for (int i = 3; i < numFeats; i++) { ptsAugFeats[i + 5] = ptsFeats[i]; }
      }
    }
  }
}

template <class T>
__global__ void scatterPtsCounts(Ref2D<int32_t> outCoors,  // [maxNumActOut, 4]
                                 Ref1D<int32_t> scatterTo,  // [numPtsIn]
                                 Ref1D<int32_t> scatterCount,  // [maxNumActOut]
                                 int32_t* numActOut,
                                 const Ref1D<int32_t> hashSlot,  // [numPtsIn]
                                 const Ref1D<int32_t> hashKey,  // [numPtsIn * 2]
                                 const Ref1D<int32_t> hashValueIncScan,  // [numPtsIn * 2]
                                 const Size<4> inSpatialShape) {
  auto numPtsIn = hashSlot.size(0);
  for (size_t ix : KernelLoopX(numPtsIn)) {
    auto maxNumActOut = outCoors.size(0);
    auto slot = hashSlot[ix];
    if (slot >= 0) {
      auto uniqueIndex = hashValueIncScan[slot] - 1;
      auto coorOffset = hashKey[slot];
      if (uniqueIndex < maxNumActOut) {
        scatterTo[ix] = uniqueIndex;
        auto first = atomicAdd(&scatterCount[uniqueIndex], 1);
        if (first == 0) { inSpatialShape.template deserialize(&outCoors(uniqueIndex, 0), coorOffset); }
      }
    }
    if (ix == 0) { *numActOut = (*numActOut) > maxNumActOut ? maxNumActOut : (*numActOut); }
  }
}

template <class Hash> __global__ void resetHashKernel(Hash hash) {
  for (size_t ix : KernelLoopX(hash.size())) { hash.keys()[ix] = hash.EMPTY; }
}
}  // namespace kernel

namespace func {
constexpr int hashSpace = 2;

size_t CylinderEncoderMalloc(const GPU& d, const size_t numPtsIn) {
  size_t reqBytes = 0;
  ssize_t numElemHash = numPtsIn * hashSpace;
  cub::DeviceScan::InclusiveSum<int32_t*, int32_t*>(nullptr, reqBytes, nullptr, nullptr, numElemHash, d.getStream());
  reqBytes += (3 * numElemHash + numPtsIn) * sizeof(int32_t);
  return reqBytes;
}

template <class T>
void CylinderEncoder(const GPU& d,
                     Ref1D<uint8_t>& workingStorage,
                     Ref2D<T>& batchPointAugFeats,
                     Ref1D<int32_t>& scatterTo,
                     Ref1D<int32_t>& scatterCount,
                     Ref2D<int32_t>& outCoors,
                     int32_t* numActOut,
                     const Ref2D<const T>& batchPointFeats,
                     const Ref1D<const int32_t>& batchIndices,
                     const Ref1D<const T>& cylinderConfig,
                     const Size<4>& inSpatialShape) {
  auto numPtsIn = batchPointFeats.size(0);
  auto maxNumActOut = batchPointAugFeats.size(0);
  ssize_t numElemHash = numPtsIn * hashSpace;
  uint8_t* workingStoragePtr = workingStorage.data();
  Ref1D<int32_t> hashKey(reinterpret_cast<int32_t*>(workingStoragePtr), {numElemHash});
  workingStoragePtr += hashKey.numby();
  Ref1D<int32_t> hashValue(reinterpret_cast<int32_t*>(workingStoragePtr), {numElemHash});
  workingStoragePtr += hashValue.numby();
  kernel::HashTable hash(hashKey.data(), hashValue.data(), numElemHash);

  Ref1D<int32_t> hashValueIncScan(reinterpret_cast<int32_t*>(workingStoragePtr), {numElemHash});
  workingStoragePtr += hashValueIncScan.numby();
  Ref1D<int32_t> hashSlot(reinterpret_cast<int32_t*>(workingStoragePtr), {numPtsIn});
  workingStoragePtr += hashSlot.numby();
  size_t workingStorageBytes = workingStorage.size(0) - (workingStoragePtr - workingStorage.data());
  {
    cudaMemsetAsync(hashSlot.data(), 0xFF, hashSlot.numby(), d.getStream());
    cudaMemsetAsync(hashValue.data(), 0x00, hashValue.numby(), d.getStream());
    cudaMemsetAsync(scatterTo.data(), 0xFF, scatterTo.numby(), d.getStream());
    cudaMemsetAsync(scatterCount.data(), 0x00, scatterCount.numby(), d.getStream());
    cudaMemsetAsync(batchPointAugFeats.data(), 0x00, batchPointAugFeats.numby(), d.getStream());
    kernel::resetHashKernel<kernel::HashTable><<<getBlocks(hash.size()), CUDA_NUM_THREADS, 0, d.getStream()>>>(hash);
    kernel::cylinderize<T, kernel::HashTable><<<getBlocks(numPtsIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        hashSlot, hash, batchPointAugFeats, batchPointFeats, batchIndices, cylinderConfig, inSpatialShape);
    cub::DeviceScan::InclusiveSum(workingStoragePtr, workingStorageBytes, hashValue.data(), hashValueIncScan.data(),
                                  hash.size(), d.getStream());
    cudaMemcpyAsync(numActOut, &hashValueIncScan[hashValueIncScan.numel() - 1], sizeof(*numActOut),
                    cudaMemcpyDeviceToDevice, d.getStream());
    kernel::scatterPtsCounts<int32_t><<<getBlocks(numPtsIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        outCoors, scatterTo, scatterCount, numActOut, hashSlot, hashKey, hashValueIncScan, inSpatialShape);
    CHECK_CUDA_ERR();
  }
}
}  // namespace func

struct CylinderEncoderPluginConsts {
  static constexpr const char* name = "CylinderEncoder";
  static constexpr const char* version = "1.0";
};

#pragma pack(push, 1)
struct CylinderEncoderPluginParam {
  int32_t maxNumActOut;
};
#pragma pack(pop)

class CylinderEncoderPlugin : public IPluginV2DynamicExt {
 private:
  CylinderEncoderPluginParam p;
  const char* mNamespace;

 public:
  CylinderEncoderPlugin() = delete;
  CylinderEncoderPlugin(const CylinderEncoderPluginParam& param) : p(param){};
  CylinderEncoderPlugin(const void* data, size_t length) {
    SerializeStream s(data, length);
    s >> p;
  }
  size_t getSerializationSize() const NOEXCEPT override { return sizeof(p); }
  void serialize(void* buffer) const NOEXCEPT override {
    SerializeStream s(buffer);
    s << p;
  }
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new CylinderEncoderPlugin(p);
    obj->setPluginNamespace(mNamespace);
    return obj;
  }
  int32_t initialize() NOEXCEPT override { return 0; };
  void terminate() NOEXCEPT override{};
  void destroy() NOEXCEPT override { delete this; };

  /**
   * IO Part:
   *    Input:
   *        0: batchPointFeats      [float]     [mMaxNumActIn, inChannels]
   *        1: batchIndices         [int32]     [mMaxNumActIn]
   *        2. cylinderConfig       [float]     [6]
   *        3: inSpatialShape       [void]      [B, 0, Z, Y, X]
   *    Output:
   *        0: outFeats             [float]     [mMaxNumActIn, inChannels + 5]
   *        1: scatterTo            [int32]     [mMaxNumActIn]
   *        2: scatterCount         [int32]     [mMaxNumActOut]
   *        3: outCoors             [int32]     [mMaxNumActOut, 4]
   *        4: numActOut            [int32]     [1]
   * */
  int32_t getNbOutputs() const NOEXCEPT override { return 5; };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    switch (outputIndex) {
    case 0: {
      return {2,
              {inputs[0].d[0],
               exprBuilder.operation(DimensionOperation::kSUM, *(inputs[0].d[1]), *(exprBuilder.constant(5)))}};
    }
    case 1: {
      return {1, {inputs[0].d[0]}};
    }
    case 2: {
      return {1, {exprBuilder.constant(p.maxNumActOut)}};
    }
    case 3: {
      return {2, {exprBuilder.constant(p.maxNumActOut), exprBuilder.constant(4)}};
    }
    default: {
      return {1, {exprBuilder.constant(1)}};
    }
    }
  }
  DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const NOEXCEPT override {
    switch (index) {
    case 0: return inputTypes[0];
    default: return DataType::kINT32;
    }
  }
  bool supportsFormatCombination(int32_t pos,
                                 const PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) NOEXCEPT override {
    switch (pos) {
    case 0: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kFLOAT);
    case 1:
    case 3:
    case 5:
    case 6:
    case 7:
    case 8: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kINT32);
    default: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == inOut[0].type);
    }
  }
  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return CylinderEncoderPluginConsts::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return CylinderEncoderPluginConsts::version; };
  void setPluginNamespace(AsciiChar const* pluginNamespace) NOEXCEPT override { mNamespace = pluginNamespace; };
  AsciiChar const* getPluginNamespace() const NOEXCEPT override { return mNamespace; };

  /**
   * Runtime Part
   * */
  void configurePlugin(const DynamicPluginTensorDesc* in,
                       int32_t nbInputs,
                       const DynamicPluginTensorDesc* out,
                       int32_t nbOutputs) NOEXCEPT override{};
  size_t getWorkspaceSize(const PluginTensorDesc* inputs,
                          int32_t nbInputs,
                          const PluginTensorDesc* outputs,
                          int32_t nbOutputs) const NOEXCEPT override {
    return func::CylinderEncoderMalloc(utils::GPU(), inputs[0].dims.d[0]);
  };
  int32_t enqueue(const PluginTensorDesc* inputDesc,
                  const PluginTensorDesc* outputDesc,
                  const void* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) NOEXCEPT override {
    /**
     * IO Part:
     *    Input:
     *        0: batchPointFeats      [float]     [mMaxNumActIn, inChannels]
     *        1: batchIndices         [int32]     [mMaxNumActIn]
     *        2. cylinderConfig       [float]     [6]
     *        3: inSpatialShape       [void]      [B, 0, Z, Y, X]
     *    Output:
     *        0: outFeats             [float]     [mMaxNumActIn, inChannels + 5]
     *        1: scatterTo            [int32]     [mMaxNumActIn]
     *        2: scatterCount         [int32]     [mMaxNumActOut]
     *        3: outCoors             [int32]     [mMaxNumActOut, 4]
     *        4: numActOut            [int32]     [1]
     * */
    Ref2D<const float> batchPointFeats = fromTensorRT<2, const float>(inputs[0], inputDesc[0]);
    Ref1D<const int32_t> batchIndices = fromTensorRT<1, const int32_t>(inputs[1], inputDesc[1]);
    Ref1D<const float> cylinderConfig = fromTensorRT<1, const float>(inputs[2], inputDesc[2]);

    Size<4> inSpatialShape{
        {inputDesc[3].dims.d[0], inputDesc[3].dims.d[2], inputDesc[3].dims.d[3], inputDesc[3].dims.d[4]}};

    Ref2D<float> outFeats = fromTensorRT<2, float>(outputs[0], outputDesc[0]);
    Ref1D<int32_t> scatterTo = fromTensorRT<1, int32_t>(outputs[1], outputDesc[1]);
    Ref1D<int32_t> scatterCount = fromTensorRT<1, int32_t>(outputs[2], outputDesc[2]);
    Ref2D<int32_t> outCoors = fromTensorRT<2, int32_t>(outputs[3], outputDesc[3]);
    int32_t* numActOut = reinterpret_cast<int32_t*>(outputs[4]);
    ssize_t workspaceSize = func::CylinderEncoderMalloc(utils::GPU(), inputDesc[0].dims.d[0]);
    Ref1D<uint8_t> workingStorage(reinterpret_cast<uint8_t*>(workspace), {workspaceSize});
    auto gpu = utils::GPU(stream);
    func::CylinderEncoder(gpu, workingStorage, outFeats, scatterTo, scatterCount, outCoors, numActOut, batchPointFeats,
                          batchIndices, cylinderConfig, inSpatialShape);
    return 0;
  };
};
class CylinderEncoderPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  CylinderEncoderPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return CylinderEncoderPluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return CylinderEncoderPluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new CylinderEncoderPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};

REGISTER_TENSORRT_PLUGIN(CylinderEncoderPluginCreator);

}  // namespace voxel_encoder
