#include "common/launch.cuh"
#include "common/refnd.h"
#include <NvInfer.h>
#include <cstring>
#include <vector>

#define NOEXCEPT noexcept

namespace spconv {
using utils::GPU;
using utils::io::SerializeStream;
using utils::launch::CUDA_NUM_THREADS;
using utils::launch::getBlocks;
using utils::launch::KernelLoopX;
using utils::nd::fromTensorRT;
using utils::nd::Ref2D;
using utils::nd::Ref5D;
using utils::nd::Size;
using utils::nd::Vec;
using namespace nvinfer1;

namespace kernel {
template <class T>
__global__ void densify(Ref5D<T> outFeatMaps,  // [N, C, Z, Y, X]
                        const Ref2D<const T> inFeats,  // [numActIn, numFeats]
                        const Ref2D<const int32_t> inCoors)  // [numActIn, 4]
{
  auto numActIn = inFeats.size(0);
  auto numFeats = inFeats.size(1);
  for (size_t ix : KernelLoopX(numActIn)) {
    auto feat = &inFeats(ix, 0);
    auto coor = &inCoors(ix, 0);
    for (int iy = 0; iy < numFeats; iy++) { outFeatMaps(coor[0], iy, coor[1], coor[2], coor[3]) = feat[iy]; }
  }
}
}  // namespace kernel

struct BEVDensifyPluginConsts {
  static constexpr const char* name = "BEVDensify";
  static constexpr const char* version = "1.0";
};

#pragma pack(push, 1)
struct BEVDensifyPluginParam {
  int32_t outChannels;
};
#pragma pack(pop)

class BEVDensifyPlugin : public IPluginV2DynamicExt {
 private:
  BEVDensifyPluginParam p;
  const char* mNamespace;

 public:
  BEVDensifyPlugin() = delete;
  BEVDensifyPlugin(const BEVDensifyPluginParam& param) : p(param){};
  BEVDensifyPlugin(const void* data, size_t length) {
    SerializeStream s(data, length);
    s >> p;
  }
  size_t getSerializationSize() const NOEXCEPT override { return sizeof(p); }
  void serialize(void* buffer) const NOEXCEPT override {
    SerializeStream s(buffer);
    s << p;
  }
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new BEVDensifyPlugin(p);
    obj->setPluginNamespace(mNamespace);
    return obj;
  }
  int32_t initialize() NOEXCEPT override { return 0; };
  void terminate() NOEXCEPT override{};
  void destroy() NOEXCEPT override { delete this; };

  /**
   * IO Part:
   *    Input:
   *        0: inFeats          [float/half]    [mMaxNumActIn, inChannels]
   *        1: inCoors          [int32]         [mMaxNumActIn, NDim + 1]
   *        2: numActIn         [int32]         [1]
   *        3: inSpatialShape   [void]          [B, 0, Z, Y, X]
   *    Output:
   *        0: outFeatMaps      [float/half]    [B, inChannels*Z, Y, X]
   * */
  int32_t getNbOutputs() const NOEXCEPT override { return 1; };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    return {4,
            {
                inputs[3].d[0],
                exprBuilder.constant(p.outChannels),
                inputs[3].d[3],
                inputs[3].d[4],
            }};
  }
  DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const NOEXCEPT override {
    return inputTypes[0];
  }
  bool supportsFormatCombination(int32_t pos,
                                 const PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) NOEXCEPT override {
    switch (pos) {
    case 0:
      return inOut[pos].format == TensorFormat::kLINEAR &&
          (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
    case 1:
    case 2:
    case 3: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kINT32);
    default: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == inOut[0].type);
    }
  }
  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return BEVDensifyPluginConsts::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return BEVDensifyPluginConsts::version; };
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
     *        0: inFeats          [float/half]    [mMaxNumActIn, inChannels]
     *        1: inCoors          [int32]         [mMaxNumActIn, NDim + 1]
     *        2: numActIn         [int32]         [1]
     *        3: inSpatialShape   [void]          [B, 0, Z, Y, X]
     *    Output:
     *        0: outFeatMaps      [float/half]    [B, inChannels*Z, Y, X]
     * */
    int32_t numActIn;
    cudaMemcpyAsync(&numActIn, inputs[2], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    int32_t inChannels = inputDesc[0].dims.d[1];
    cudaStreamSynchronize(stream);
    Ref2D<const int32_t> inCoors(reinterpret_cast<const int32_t*>(inputs[1]), {numActIn, 4});
    if (inputDesc[0].type == DataType::kHALF) {
      using T = half;
      Ref2D<const T> inFeats(reinterpret_cast<const T*>(inputs[0]), {numActIn, inChannels});
      Ref5D<T> outFeatMaps(
          reinterpret_cast<T*>(outputs[0]),
          {inputDesc[3].dims.d[0], inChannels, inputDesc[3].dims.d[2], inputDesc[3].dims.d[3], inputDesc[3].dims.d[4]});
      cudaMemsetAsync(outFeatMaps.data(), 0x00, outFeatMaps.numby(), stream);
      if (numActIn > 0)
        kernel::densify<<<getBlocks(numActIn), CUDA_NUM_THREADS, 0, stream>>>(outFeatMaps, inFeats, inCoors);
    } else {
      using T = float;
      Ref2D<const T> inFeats(reinterpret_cast<const T*>(inputs[0]), {numActIn, inChannels});
      Ref5D<T> outFeatMaps(
          reinterpret_cast<T*>(outputs[0]),
          {inputDesc[3].dims.d[0], inChannels, inputDesc[3].dims.d[2], inputDesc[3].dims.d[3], inputDesc[3].dims.d[4]});
      cudaMemsetAsync(outFeatMaps.data(), 0x00, outFeatMaps.numby(), stream);
      if (numActIn > 0)
        kernel::densify<<<getBlocks(numActIn), CUDA_NUM_THREADS, 0, stream>>>(outFeatMaps, inFeats, inCoors);
    }
    return 0;
  };
};
class BEVDensifyPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  BEVDensifyPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return BEVDensifyPluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return BEVDensifyPluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new BEVDensifyPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};
REGISTER_TENSORRT_PLUGIN(BEVDensifyPluginCreator);
}  // namespace spconv
