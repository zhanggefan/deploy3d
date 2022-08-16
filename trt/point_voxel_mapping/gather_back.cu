#include "common/launch.cuh"
#include "common/macros.h"
#include "common/refnd.h"
#include "cuda_fp16.hpp"
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

template <class T>
__global__ void gatherFeats(Ref2D<T> outFeats,  // [numPtsIn, P]
                            const Ref2D<const T> inFeats,  // [maxNumActOut, P]
                            const Ref1D<const int32_t> scatterIndex)  // [numPtsIn]
{
  auto numPtsIn = outFeats.size(0);
  auto numFeats = outFeats.size(1);
  for (size_t ix : KernelLoopX(numPtsIn)) {
    int voxelIdx = scatterIndex(ix);
    if (voxelIdx < 0) continue;
    for (int j = 0; j < numFeats; j++) { outFeats(ix, j) = inFeats(voxelIdx, j); }
  }
}
}  // namespace kernel

namespace func {
template <class T>
void gatherBack(const GPU& d,
                Ref2D<T> outFeats,
                const Ref2D<const T> inFeats,
                const Ref1D<const int32_t> scatterIndex) {
  auto numPtsIn = outFeats.size(0);
  kernel::gatherFeats<<<getBlocks(numPtsIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, inFeats, scatterIndex);
}
}  // namespace func

struct GatherBackPluginConsts {
  static constexpr const char* name = "GatherBack";
  static constexpr const char* version = "2.0";
};

class GatherBackPlugin : public IPluginV2DynamicExt {
 private:
  const char* mNamespace;

 public:
  GatherBackPlugin() = delete;
  GatherBackPlugin(const void* data, size_t length){};
  size_t getSerializationSize() const NOEXCEPT override { return 0; }
  void serialize(void* buffer) const NOEXCEPT override{};
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new GatherBackPlugin(nullptr, 0);
    obj->setPluginNamespace(mNamespace);
    return obj;
  }
  int32_t initialize() NOEXCEPT override { return 0; };
  void terminate() NOEXCEPT override{};
  void destroy() NOEXCEPT override { delete this; };

  /**
   * IO Part:
   *    Input:
   *        0: reducedFeats         [float]     [mMaxNumActOut, inChannels]
   *        1: scatterTo            [int32]     [mMaxNumActIn]
   *    Output:
   *        0: batchPointFeats      [float]     [mMaxNumActIn, inChannels]
   * */
  int32_t getNbOutputs() const NOEXCEPT override { return 1; };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    return {2, {inputs[1].d[0], inputs[0].d[1]}};
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
    case 2: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == inOut[0].type);
    default: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kINT32);
    }
  }
  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return GatherBackPluginConsts::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return GatherBackPluginConsts::version; };
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
     *        0: reducedFeats         [float]     [mMaxNumActOut, inChannels]
     *        1: scatterTo            [int32]     [mMaxNumActIn]
     *    Output:
     *        0: batchPointFeats      [float]     [mMaxNumActIn, inChannels]
     * */
    Ref2D<const float> inFeats = fromTensorRT<2, const float>(inputs[0], inputDesc[0]);
    Ref1D<const int32_t> scatterIndex = fromTensorRT<1, const int32_t>(inputs[1], inputDesc[1]);
    Ref2D<float> outFeats = fromTensorRT<2, float>(outputs[0], outputDesc[0]);
    func::gatherBack<float>(utils::GPU(), outFeats, inFeats, scatterIndex);
    return 0;
  };
};

class GatherBackPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  GatherBackPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return GatherBackPluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return GatherBackPluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new GatherBackPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};
REGISTER_TENSORRT_PLUGIN(GatherBackPluginCreator);
}  // namespace point_voxel_mapping