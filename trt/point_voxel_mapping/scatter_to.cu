#include "common/launch.cuh"
#include "common/macros.h"
#include "common/refnd.h"
#include <NvInfer.h>
#include <cstring>
#include <vector>

#define NOEXCEPT noexcept

namespace point_voxel_mapping {
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

DEVICE_INLINE static void atomicMax(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old || __int_as_float(old) < val);
}

DEVICE_INLINE static void atomicMax(half* address, half val) {
  auto* address_as_s = reinterpret_cast<uint16_t*>(address);
  uint16_t old = *address_as_s, assumed;
  do {
    assumed = old;
    half hassumed = __ushort_as_half(assumed);
    old = atomicCAS(address_as_s, assumed, __half_as_ushort(__hgt(val, hassumed) ? val : hassumed));
  } while (assumed != old || __ushort_as_half(old) < val);
}

template <class T>
__global__ void maxReduceFeats(Ref2D<T> outFeats,  // [maxNumActOut, P]
                               const Ref2D<const T> inFeats,  // [numPtsIn, P]
                               const Ref1D<const int32_t> scatterIndex)  // [numPtsIn]
{
  auto numPtsIn = inFeats.size(0);
  auto numFeats = inFeats.size(1);

  for (size_t ix : KernelLoopX(numPtsIn)) {
    int voxelIdx = scatterIndex(ix);
    if (voxelIdx < 0) continue;
    for (int j = 0; j < numFeats; j++) { atomicMax(&outFeats(voxelIdx, j), inFeats(ix, j)); }
  }
}

template <class T>
__global__ void addReduceFeats(Ref2D<T> outFeats,  // [maxNumActOut, P]
                               const Ref2D<const T> inFeats,  // [numPtsIn, P]
                               const Ref1D<const int32_t> scatterIndex)  // [numPtsIn]
{
  auto numPtsIn = inFeats.size(0);
  auto numFeats = inFeats.size(1);

  for (size_t ix : KernelLoopX(numPtsIn)) {
    int voxelIdx = scatterIndex(ix);
    if (voxelIdx < 0) continue;
    for (int j = 0; j < numFeats; j++) { atomicAdd(&outFeats(voxelIdx, j), inFeats(ix, j)); }
  }
}

template <class T> __global__ void meanFeats(Ref2D<T> outFeats, const Ref1D<const int32_t> scatterCount) {
  auto maxNumActOut = scatterCount.size(0);
  auto numFeats = outFeats.size(1);
  for (size_t ix : KernelLoopX(maxNumActOut)) {
    int numVoxelPts = scatterCount(ix);
    if (numVoxelPts <= 0) continue;
    for (int j = 0; j < numFeats; j++) { outFeats(ix, j) /= numVoxelPts; }
  }
}

template <class T> __global__ void initFeats(Ref2D<T> outFeats, const T initVal) {
  auto maxNumActOut = outFeats.size(0);
  auto numFeats = outFeats.size(1);
  for (size_t ix : KernelLoopX(maxNumActOut)) {
    for (int j = 0; j < numFeats; j++) { outFeats(ix, j) = initVal; }
  }
}
}  // namespace kernel

namespace func {
template <class T>
void scatterTo(const GPU& d,
               Ref2D<T> outFeats,
               const Ref2D<const T> inFeats,
               const Ref1D<const int32_t> scatterIndex,
               const Ref1D<const int32_t> scatterCount,
               const int8_t reduceType) {
  auto numPtsIn = inFeats.size(0);
  auto maxNumActOut = outFeats.size(0);
  if (reduceType == 0)  // max reduce
  {
    kernel::initFeats<T>
        <<<getBlocks(maxNumActOut), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, std::numeric_limits<T>::lowest());
    kernel::maxReduceFeats<<<getBlocks(numPtsIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, inFeats,
                                                                                        scatterIndex);
  } else {  // mean reduce
    kernel::initFeats<T><<<getBlocks(maxNumActOut), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, 0);
    kernel::addReduceFeats<<<getBlocks(numPtsIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, inFeats,
                                                                                        scatterIndex);
    kernel::meanFeats<<<getBlocks(maxNumActOut), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, scatterCount);
  }
}
}  // namespace func

struct ScatterToPluginConsts {
  static constexpr const char* name = "ScatterTo";
  static constexpr const char* version = "1.0";
};

#pragma pack(push, 1)
struct ScatterToPluginParam {
  int8_t reduceType;
};
#pragma pack(pop)

class ScatterToPlugin : public IPluginV2DynamicExt {
 private:
  ScatterToPluginParam p;
  const char* mNamespace;

 public:
  ScatterToPlugin() = delete;
  ScatterToPlugin(const ScatterToPluginParam& param) : p(param){};
  ScatterToPlugin(const void* data, size_t length) {
    SerializeStream s(data, length);
    s >> p;
  }
  size_t getSerializationSize() const NOEXCEPT override { return sizeof(p); }
  void serialize(void* buffer) const NOEXCEPT override {
    SerializeStream s(buffer);
    s << p;
  }
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new ScatterToPlugin(p);
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
   *        1: scatterTo            [int32]     [mMaxNumActIn]
   *        2: scatterCount         [int32]     [mMaxNumActOut]
   *    Output:
   *        0: reducedFeats         [float]     [mMaxNumActOut, inChannels]
   * */
  int32_t getNbOutputs() const NOEXCEPT override { return 1; };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    return {2, {inputs[2].d[0], inputs[0].d[1]}};
  }
  DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const NOEXCEPT override {
    return inputTypes[0];
  }
  bool supportsFormatCombination(int32_t pos,
                                 const PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) NOEXCEPT override {
    switch (pos) {
    case 0: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kFLOAT);
    case 3: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == inOut[0].type);
    default: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kINT32);
    }
  }
  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return ScatterToPluginConsts::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return ScatterToPluginConsts::version; };
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
    return 0;
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
     *        1: scatterTo            [int32]     [mMaxNumActIn]
     *        2: scatterCount         [int32]     [mMaxNumActOut]
     *    Output:
     *        0: reducedFeats         [float]     [mMaxNumActOut, inChannels]
     * */
    Ref2D<const float> inFeats = fromTensorRT<2, const float>(inputs[0], inputDesc[0]);
    Ref1D<const int32_t> scatterIndex = fromTensorRT<1, const int32_t>(inputs[1], inputDesc[1]);
    Ref1D<const int32_t> scatterCount = fromTensorRT<1, const int32_t>(inputs[2], inputDesc[2]);
    Ref2D<float> outFeats = fromTensorRT<2, float>(outputs[0], outputDesc[0]);
    func::scatterTo<float>(utils::GPU(), outFeats, inFeats, scatterIndex, scatterCount, p.reduceType);
    return 0;
  };
};

class ScatterToPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  ScatterToPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return ScatterToPluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return ScatterToPluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new ScatterToPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};
REGISTER_TENSORRT_PLUGIN(ScatterToPluginCreator);
}  // namespace point_voxel_mapping