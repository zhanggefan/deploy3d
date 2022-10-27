#include "common/cub.cuh"
#include "common/hash32.h"
#include "common/launch.cuh"
#include "common/macros.h"
#include "common/refnd.h"
#include <NvInfer.h>
#include <cstring>
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
__global__ void voxelize(Ref1D<int32_t> hashSlot,  // [numPtsIn]
                         Hash hash,
                         const Ref2D<const T> batchPointFeats,  // [numPtsIn, 4]
                         const Ref1D<const int32_t> batchIndices,  // [numPtsIn]
                         const Ref1D<const T> voxelConfig,  // [6]
                         const Size<4> inSpatialShape) {
  auto numPtsIn = batchPointFeats.size(0);
  for (size_t ix : KernelLoopX(numPtsIn)) {
    const T* ptsFeats = &batchPointFeats(ix, 0);
    const auto numFeats = batchPointFeats.size(1);
    bool notNan = true;
    for (int i = 0; i < numFeats; i++) { notNan = notNan && isfinite(ptsFeats[i]); }
    if (notNan) {
      Vec<4, int32_t> ptsCoors{};
      ptsCoors[0] = batchIndices[ix];
      if (ptsCoors[0] >= 0) {
        T xMin = voxelConfig[0];
        T yMin = voxelConfig[1];
        T zMin = voxelConfig[2];
        T xSize = voxelConfig[3];
        T ySize = voxelConfig[4];
        T zSize = voxelConfig[5];
        ptsCoors[3] = static_cast<int32_t>((ptsFeats[0] - xMin) / xSize);
        ptsCoors[2] = static_cast<int32_t>((ptsFeats[1] - yMin) / ySize);
        ptsCoors[1] = static_cast<int32_t>((ptsFeats[2] - zMin) / zSize);
        if (inSpatialShape.template is_valid(ptsCoors)) {
          hashSlot[ix] = hash.insert(inSpatialShape.template offset(ptsCoors), 1);
        }
      }
    }
  }
}

template <class T>
__global__ void scatterAddFeats(Ref2D<T> outFeats,  // [maxNumActOut, 4]
                                Ref2D<int32_t> outCoors,  // [maxNumActOut, 4]
                                Ref1D<int32_t> outPtsCount,  // [maxNumActOut]
                                int32_t* numActOut,
                                const Ref2D<const T> batchPointFeats,  // [numPtsIn, 4]
                                const Ref1D<int32_t> hashSlot,  // [numPtsIn]
                                const Ref1D<int32_t> hashKey,  // [numPtsIn * 2]
                                const Ref1D<int32_t> hashValueIncScan,  // [numPtsIn * 2]
                                const Size<4> inSpatialShape) {
  auto numPtsIn = batchPointFeats.size(0);
  for (size_t ix : KernelLoopX(numPtsIn)) {
    auto maxNumActOut = outFeats.size(0);
    auto slot = hashSlot[ix];
    if (slot >= 0) {
      auto uniqueIndex = hashValueIncScan[slot] - 1;
      auto coorOffset = hashKey[slot];
      if (uniqueIndex < maxNumActOut) {
        auto first = atomicAdd(&outPtsCount[uniqueIndex], 1);
        auto numFeats = outFeats.size(1);
        for (int iy = 0; iy < numFeats; iy++) {
          auto feat = batchPointFeats(ix, iy);
          atomicAdd(&outFeats(uniqueIndex, iy), feat);
        }
        if (first == 0) { inSpatialShape.template deserialize(&outCoors(uniqueIndex, 0), coorOffset); }
      }
    }
    if (ix == 0) { *numActOut = (*numActOut) > maxNumActOut ? maxNumActOut : (*numActOut); }
  }
}

template <class T>
__global__ void meanFeats(Ref2D<T> outFeats,  // [maxNumActOut, 4]
                          const Ref1D<int32_t> outPtsCount,  // [maxNumActOut]
                          const int32_t* numActOut) {
  for (size_t ix : KernelLoopX(*numActOut)) {
    auto numFeats = outFeats.size(1);
    auto count = static_cast<T>(outPtsCount[ix]);
    for (int iy = 0; iy < numFeats; iy++) { outFeats(ix, iy) = outFeats(ix, iy) / count; }
  }
}

template <class Hash> __global__ void resetHashKernel(Hash hash) {
  for (size_t ix : KernelLoopX(hash.size())) { hash.keys()[ix] = hash.EMPTY; }
}
}  // namespace kernel

namespace func {
constexpr int hashSpace = 2;

size_t simpleMeanEncoderMalloc(const GPU& d, const size_t numPtsIn) {
  size_t reqBytes = 0;
  ssize_t numElemHash = numPtsIn * hashSpace;
  DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::InclusiveSum<int32_t*, int32_t*>(nullptr, reqBytes, nullptr, nullptr,
                                                                               numElemHash, d.getStream());
  reqBytes += (3 * numElemHash + 2 * numPtsIn) * sizeof(int32_t);
  return reqBytes;
}

template <class T>
void simpleMeanEncoder(const GPU& d,
                       Ref1D<uint8_t>& workingStorage,
                       Ref2D<T>& outFeats,
                       Ref2D<int32_t>& outCoors,
                       int32_t* numActOut,
                       const Ref2D<const T>& batchPointFeats,
                       const Ref1D<const int32_t>& batchIndices,
                       const Ref1D<const T>& voxelConfig,
                       const Size<4>& inSpatialShape) {
  auto numPtsIn = batchPointFeats.size(0);
  auto maxNumActOut = outFeats.size(0);
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
  Ref1D<int32_t> outPtsCount(reinterpret_cast<int32_t*>(workingStoragePtr), {numPtsIn});
  workingStoragePtr += outPtsCount.numby();
  size_t workingStorageBytes = workingStorage.size(0) - (workingStoragePtr - workingStorage.data());

  {
    cudaMemsetAsync(hashSlot.data(), 0xFF, hashSlot.numby(), d.getStream());
    cudaMemsetAsync(hashValue.data(), 0x00, hashValue.numby(), d.getStream());
    cudaMemsetAsync(outPtsCount.data(), 0x00, outPtsCount.numby(), d.getStream());
    cudaMemsetAsync(outFeats.data(), 0x00, outFeats.numby(), d.getStream());
    kernel::resetHashKernel<kernel::HashTable><<<getBlocks(hash.size()), CUDA_NUM_THREADS, 0, d.getStream()>>>(hash);
    kernel::voxelize<<<getBlocks(numPtsIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        hashSlot, hash, batchPointFeats, batchIndices, voxelConfig, inSpatialShape);
    DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::InclusiveSum(workingStoragePtr, workingStorageBytes, hashValue.data(),
                                                             hashValueIncScan.data(), hash.size(), d.getStream());
    cudaMemcpyAsync(numActOut, &hashValueIncScan[hashValueIncScan.numel() - 1], sizeof(*numActOut),
                    cudaMemcpyDeviceToDevice, d.getStream());
    kernel::scatterAddFeats<<<getBlocks(numPtsIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        outFeats, outCoors, outPtsCount, numActOut, batchPointFeats, hashSlot, hashKey, hashValueIncScan,
        inSpatialShape);
    kernel::meanFeats<<<getBlocks(std::min(maxNumActOut, numPtsIn)), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        outFeats, outPtsCount, numActOut);
    CHECK_CUDA_ERR();
  }
}
}  // namespace func

struct SimpleMeanEncoderPluginConsts {
  static constexpr const char* name = "SimpleMeanEncoder";
  static constexpr const char* version = "2.0";
};

#pragma pack(push, 1)
struct SimpleMeanEncoderPluginParam {
  int32_t maxNumActOut;
};
#pragma pack(pop)

class SimpleMeanEncoderPlugin : public IPluginV2DynamicExt {
 private:
  SimpleMeanEncoderPluginParam p;
  const char* mNamespace;

 public:
  SimpleMeanEncoderPlugin() = delete;
  SimpleMeanEncoderPlugin(const SimpleMeanEncoderPluginParam& param) : p(param){};
  SimpleMeanEncoderPlugin(const void* data, size_t length) {
    SerializeStream s(data, length);
    s >> p;
  }
  size_t getSerializationSize() const NOEXCEPT override { return sizeof(p); }
  void serialize(void* buffer) const NOEXCEPT override {
    SerializeStream s(buffer);
    s << p;
  }
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new SimpleMeanEncoderPlugin(p);
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
   *        2. voxelConfig          [float]     [6]
   *        3: inSpatialShape       [void]      [B, 0, Z, Y, X]
   *    Output:
   *        0: outFeats             [float]     [mMaxNumActOut, inChannels]
   *        1: outCoors             [int32]     [mMaxNumActOut, 4]
   *        2: numActOut            [int32]     [1]
   * */
  int32_t getNbOutputs() const NOEXCEPT override { return 3; };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    switch (outputIndex) {
    case 0: {
      DimsExprs outFeats(inputs[0]);
      outFeats.d[0] = exprBuilder.constant(p.maxNumActOut);
      return outFeats;
    }
    case 1: {
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
    case 6: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kINT32);
    default: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == inOut[0].type);
    }
  }
  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return SimpleMeanEncoderPluginConsts::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return SimpleMeanEncoderPluginConsts::version; };
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
    return func::simpleMeanEncoderMalloc(utils::GPU(), inputs[0].dims.d[0]);
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
     *        2. voxelConfig          [float]     [6]
     *        3: inSpatialShape       [void]      [B, 0, Z, Y, X]
     *    Output:
     *        0: outFeats             [float]     [mMaxNumActOut, inChannels]
     *        1: outCoors             [int32]     [mMaxNumActOut, 4]
     *        2: numActOut            [int32]     [1]
     * */
    Ref2D<const float> batchPointFeats = fromTensorRT<2, const float>(inputs[0], inputDesc[0]);
    Ref1D<const int32_t> batchIndices = fromTensorRT<1, const int32_t>(inputs[1], inputDesc[1]);
    Ref1D<const float> voxelConfig = fromTensorRT<1, const float>(inputs[2], inputDesc[2]);

    Size<4> inSpatialShape{
        {inputDesc[3].dims.d[0], inputDesc[3].dims.d[2], inputDesc[3].dims.d[3], inputDesc[3].dims.d[4]}};

    Ref2D<float> outFeats = fromTensorRT<2, float>(outputs[0], outputDesc[0]);
    Ref2D<int32_t> outCoors = fromTensorRT<2, int32_t>(outputs[1], outputDesc[1]);
    int32_t* numActOut = reinterpret_cast<int32_t*>(outputs[2]);
    ssize_t workspaceSize = func::simpleMeanEncoderMalloc(utils::GPU(), inputDesc[0].dims.d[0]);
    Ref1D<uint8_t> workingStorage(reinterpret_cast<uint8_t*>(workspace), {workspaceSize});
    auto gpu = utils::GPU(stream);
    func::simpleMeanEncoder(gpu, workingStorage, outFeats, outCoors, numActOut, batchPointFeats, batchIndices,
                            voxelConfig, inSpatialShape);
    return 0;
  };
};
class SimpleMeanEncoderPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  SimpleMeanEncoderPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return SimpleMeanEncoderPluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return SimpleMeanEncoderPluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new SimpleMeanEncoderPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};

REGISTER_TENSORRT_PLUGIN(SimpleMeanEncoderPluginCreator);

}  // namespace voxel_encoder
