#include "common/refnd.h"
#include "geometry.h"
#include "index.h"
#include <NvInfer.h>
#include <cstring>
#include <string>
#include <vector>

#define NOEXCEPT noexcept

namespace spconv {
using utils::io::SerializeStream;
using utils::nd::fromTensorRT;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;
using utils::nd::Size;
using utils::nd::Vec;
using namespace nvinfer1;

template <size_t NDim> struct SPConvIdxPluginConsts;
template <> struct SPConvIdxPluginConsts<1> {
  static constexpr const char* name = "SPConvIdx1d";
  static constexpr const char* version = "2.0";
};
template <> struct SPConvIdxPluginConsts<2> {
  static constexpr const char* name = "SPConvIdx2d";
  static constexpr const char* version = "2.0";
};
template <> struct SPConvIdxPluginConsts<3> {
  static constexpr const char* name = "SPConvIdx3d";
  static constexpr const char* version = "2.0";
};
template <> struct SPConvIdxPluginConsts<4> {
  static constexpr const char* name = "SPConvIdx4d";
  static constexpr const char* version = "2.0";
};

#pragma pack(push, 1)
template <size_t NDim> struct SPConvIdxPluginParam {
  Vec<NDim, int32_t> kernelSize;
  Vec<NDim, int32_t> stride;
  Vec<NDim, int32_t> padding;
  Vec<NDim, int32_t> dilation;
  Vec<NDim, int32_t> outPadding;
  int32_t maxNumActOut;
  bool subM;
  bool transpose;
};
#pragma pack(pop)

template <size_t NDim> class SPConvIdxPlugin : public IPluginV2DynamicExt {
 public:
 private:
  SPConvIdxPluginParam<NDim> p;
  const char* mNamespace;

 public:
  static constexpr size_t NumDim = NDim;

  /**
   * Lifecycle Part
   * */
  SPConvIdxPlugin() = delete;
  SPConvIdxPlugin(const SPConvIdxPluginParam<NDim>& param) : p(param) {}
  SPConvIdxPlugin(const void* data, size_t length) {
    SerializeStream s(data, length);
    s >> p;
  }
  size_t getSerializationSize() const NOEXCEPT override { return sizeof(p); }
  void serialize(void* buffer) const NOEXCEPT override {
    SerializeStream s(buffer);
    s << p;
  }
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new SPConvIdxPlugin<NDim>(p);
    obj->setPluginNamespace(mNamespace);
    return obj;
  }
  int32_t initialize() NOEXCEPT override { return 0; };
  void terminate() NOEXCEPT override{};
  void destroy() NOEXCEPT override { delete this; };

  /**
   * IO Part:
   *    Input:
   *        0: inCoors              [int32] [mMaxNumActIn, NDim + 1]
   *        1: numActIn             [int32] [1]
   *        2: inSpatialShape       [void]  [B, 0, Z, Y, X]  // dynamic shape
   *    Output:
   *        0: index                [int32] [3, kVol * (mMaxNumActIn + 128)]
   *        1: numIndex             [int32] [1]
   *        2: outCoors             [int32] [mMaxNumActOut, NDim + 1]
   *        3: numActOut            [int32] [1]
   *        4: outSpatialShape      [void]  [B, 0, Z, Y, X]  // dynamic shape
   * */
  int32_t getNbOutputs() const NOEXCEPT override {
    if (p.subM) return 2;
    return 5;
  };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    switch (outputIndex) {
    case 0: {  // 0: index [int32] [3, kVol * (mMaxNumActIn + 128)]
      int32_t kVol = 1;
#pragma unroll
      for (auto s : p.kernelSize) { kVol *= s; }
      return {2,
              {exprBuilder.constant(3),
               exprBuilder.operation(
                   DimensionOperation::kPROD,
                   *exprBuilder.operation(DimensionOperation::kSUM, *inputs[0].d[0], *exprBuilder.constant(128)),
                   *exprBuilder.constant(kVol))}};
    }
    case 1: {  // 1: numIndex [int32] [1]
      return {1, {exprBuilder.constant(1)}};
    }
    case 2:  // 2: outCoors [int32] [mMaxNumActOut, NDim + 1]
    {
      return {2, {exprBuilder.constant(p.maxNumActOut), exprBuilder.constant(NDim + 1)}};
    }
    case 3:  // 3: numActOut [int32] [1]
    {
      return {1, {exprBuilder.constant(1)}};
    }
    default: {  // 4: outSpatialShape [void] [B, 0, Z, Y, X]
      DimsExprs outSpatialShape(inputs[2]);
      if (p.transpose) {
#pragma unroll
        for (int i = 0; i < NDim; i++) {
          outSpatialShape.d[i + 2] =
              exprBuilder.operation(DimensionOperation::kSUM,
                                    *exprBuilder.operation(DimensionOperation::kPROD, *outSpatialShape.d[i + 2],
                                                           *exprBuilder.constant(p.stride[i])),
                                    *exprBuilder.constant(-p.stride[i] - 2 * p.padding[i] +
                                                          (p.kernelSize[i] - 1) * p.dilation[i] + p.outPadding[i] + 1));
        }
      } else {
        for (int i = 0; i < NDim; i++) {
          outSpatialShape.d[i + 2] = exprBuilder.operation(
              DimensionOperation::kCEIL_DIV,
              *exprBuilder.operation(DimensionOperation::kSUM, *outSpatialShape.d[i + 2],
                                     *exprBuilder.constant(2 * p.padding[i] - p.dilation[i] * (p.kernelSize[i] - 1))),
              *exprBuilder.constant(p.stride[i]));
        }
      }
      return outSpatialShape;
    }
    }
  }
  DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const NOEXCEPT override {
    return DataType::kINT32;
  }
  bool supportsFormatCombination(int32_t pos,
                                 const PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) NOEXCEPT override {
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT32;
  }

  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return SPConvIdxPluginConsts<NDim>::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return SPConvIdxPluginConsts<NDim>::version; };
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
    if (p.subM)
      return spconv::func::createSparseSubMIndexMalloc<NDim, int32_t>(utils::GPU(), inputs[0].dims.d[0], p.kernelSize,
                                                                      true);
    return spconv::func::createSparseConvIndexMalloc<NDim, int32_t>(utils::GPU(), inputs[0].dims.d[0], p.kernelSize,
                                                                    true);
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
     *        0: inCoors              [int32] [mMaxNumActIn, NDim + 1]
     *        1: numActIn             [int32] [1]
     *        2: inSpatialShape       [void]  [B, 0, Z, Y, X]  // dynamic shape
     *    Output:
     *        0: index                [int32] [3, kVol * (mMaxNumActIn + 128)]
     *        1: numIndex             [int32] [1]
     *        2: outCoors             [int32] [mMaxNumActOut, NDim + 1]
     *        3: numActOut            [int32] [1]
     *        4: outSpatialShape      [void]  [B, 0, Z, Y, X]  // dynamic shape
     * */
    int32_t numActIn;
    cudaMemcpyAsync(&numActIn, inputs[1], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    Ref2D<int32_t> inCoors(reinterpret_cast<int32_t*>(const_cast<void*>(inputs[0])), {numActIn, NDim + 1});
    auto index = fromTensorRT<2, int32_t>(outputs[0], outputDesc[0]);
    auto gatherIn = index.template subview(0);
    auto scatterOut = index.template subview(1);
    auto kernelOffset = index.template subview(2);
    auto numIndex = reinterpret_cast<int32_t*>(outputs[1]);
    typename Size<NDim + 1>::index_vec_t outSpatialShapeVec;
    decltype(&inputDesc[2].dims.d[0]) outSpatialShapeDims;
    if (p.subM) {
      outSpatialShapeDims = inputDesc[2].dims.d;
    } else {
      outSpatialShapeDims = outputDesc[4].dims.d;
    }
    outSpatialShapeVec[0] = outSpatialShapeDims[0];
#pragma unroll
    for (int i = 0; i < NDim; i++) outSpatialShapeVec[i + 1] = outSpatialShapeDims[i + 2];
    Size<NDim + 1> outSpatialShape(outSpatialShapeVec);
    auto gpu = utils::GPU(stream);

    if (p.subM) {
      ssize_t workspaceSize =
          spconv::func::createSparseSubMIndexMalloc<NDim, int32_t>(utils::GPU(), numActIn, p.kernelSize);
      Ref1D<uint8_t> workingStorage(reinterpret_cast<uint8_t*>(workspace), {workspaceSize});
      spconv::func::createSparseSubMIndex<NDim, int32_t>(gpu, workingStorage, gatherIn, scatterOut, kernelOffset,
                                                         numIndex, inCoors, p.kernelSize, p.stride, p.padding,
                                                         p.dilation, outSpatialShape);
      return 0;
    } else {
      auto outCoors = fromTensorRT<2, int32_t>(outputs[2], outputDesc[2]);
      auto numActOut = reinterpret_cast<int32_t*>(outputs[3]);
      ssize_t workspaceSize =
          spconv::func::createSparseConvIndexMalloc<NDim, int32_t>(utils::GPU(), numActIn, p.kernelSize);
      Ref1D<uint8_t> workingStorage(reinterpret_cast<uint8_t*>(workspace), {workspaceSize});
      spconv::func::createSparseConvIndex<NDim, int32_t>(
          gpu, workingStorage, gatherIn, scatterOut, kernelOffset, outCoors, numIndex, numActOut, inCoors, p.kernelSize,
          p.stride, p.padding, p.dilation, outSpatialShape, p.transpose, p.maxNumActOut);
      return 0;
    }
  }
};

template <size_t NDim> class SPConvIdxPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  SPConvIdxPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return SPConvIdxPluginConsts<NDim>::name; };

  const char* getPluginVersion() const NOEXCEPT override { return SPConvIdxPluginConsts<NDim>::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new SPConvIdxPlugin<NDim>(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};

using SPConvIdx1dPluginCreator = SPConvIdxPluginCreator<1>;
using SPConvIdx2dPluginCreator = SPConvIdxPluginCreator<2>;
using SPConvIdx3dPluginCreator = SPConvIdxPluginCreator<3>;
using SPConvIdx4dPluginCreator = SPConvIdxPluginCreator<4>;

REGISTER_TENSORRT_PLUGIN(SPConvIdx1dPluginCreator);
REGISTER_TENSORRT_PLUGIN(SPConvIdx2dPluginCreator);
REGISTER_TENSORRT_PLUGIN(SPConvIdx3dPluginCreator);
REGISTER_TENSORRT_PLUGIN(SPConvIdx4dPluginCreator);

}  // namespace spconv
