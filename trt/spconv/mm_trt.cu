#include "common/refnd.h"
#include "mm.h"
#include <NvInfer.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define NOEXCEPT noexcept

namespace spconv {
using utils::io::SerializeStream;
using utils::mem::DeviceVector;
using utils::nd::fromTensorRT;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;
using utils::nd::Size;
using utils::nd::Vec;
using namespace nvinfer1;

struct SPConvMMPluginConsts {
  static constexpr const char* name = "SPConvMM";
  static constexpr const char* version = "2.0";
};

#pragma pack(push, 1)
struct SPConvMMPluginParam {
  int32_t kernelVol;
  int32_t inChannels;
  int32_t outChannels;
  int32_t maxNumActOut;
  bool subM;  // deprecated
  bool inverse;
  bool withBias;
};
#pragma pack(pop)

class SPConvMMPlugin : public IPluginV2DynamicExt {
 public:
 private:
  SPConvMMPluginParam p;
  std::vector<float> wb;
  std::shared_ptr<DeviceVector> wRt;
  std::shared_ptr<DeviceVector> bRt;
  const char* mNamespace;
  using MMOp1 = spconv::mm::TensorOp<cutlass::half_t, 64, 64, 16, 32, 32, 16, cutlass::arch::Sm75>;
  using MMOp2 = spconv::mm::TensorOp<cutlass::half_t, 64, 64, 32, 16, 32, 32, cutlass::arch::Sm75>;
  using MMOp3 = spconv::mm::Simt<cutlass::half_t, 32, 64, 8, 16, 32, 8, cutlass::arch::Sm80>;

 public:
  /**
   * Lifecycle Part
   * */
  SPConvMMPlugin() = delete;
  SPConvMMPlugin(const SPConvMMPluginParam& param,
                 const std::vector<float>& weightAndBias,
                 const std::shared_ptr<DeviceVector>& wRuntime,
                 const std::shared_ptr<DeviceVector>& bRuntime)
      : p(param), wb(weightAndBias), wRt(wRuntime), bRt(bRuntime) {}
  SPConvMMPlugin(const void* data, size_t length) {
    SerializeStream s(data, length);
    s >> p;
    wb.resize((length - s.curPos()) / sizeof(float));
    s.loadRange(wb.begin(), wb.end());
    wRt = std::make_shared<DeviceVector>(p.kernelVol * p.inChannels * p.outChannels * sizeof(float));
    bRt = std::make_shared<DeviceVector>(p.outChannels * sizeof(float));
  }
  size_t getSerializationSize() const NOEXCEPT override { return sizeof(p) + wb.size() * sizeof(wb[0]); }
  void serialize(void* buffer) const NOEXCEPT override {
    SerializeStream s(buffer);
    s << p;
    s.dumpRange(wb.begin(), wb.end());
  }
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new SPConvMMPlugin(p, wb, wRt, bRt);
    obj->setPluginNamespace(mNamespace);
    return obj;
  }
  int32_t initialize() NOEXCEPT override { return 0; };
  void terminate() NOEXCEPT override{};
  void destroy() NOEXCEPT override { delete this; };

  /**
   * IO Part:
   *    Input:
   *        0: inFeats              [float/half]    [mMaxNumActIn, inChannels]
   *        1: numActIn             [int32]         [1]
   *        2: numActOut            [int32]         [1]
   *        3: index                [int32]         [3, kVol * (mMaxNumActIn + 128)]
   *        4: numIndex             [int32]         [1]
   *    Output:
   *        0: outFeats             [float/half]    [mMaxNumActOut,
   * outChannels]
   * */
  int32_t getNbOutputs() const NOEXCEPT override { return 1; };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    return {2, {exprBuilder.constant(p.maxNumActOut), exprBuilder.constant(p.outChannels)}};
  }
  DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const NOEXCEPT override {
    return inputTypes[0];
  }
  bool supportsFormatCombination(int32_t pos,
                                 const PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) NOEXCEPT override {
    switch (pos) {
    case 0: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == DataType::kHALF);
    case 5: return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == inOut[0].type);
    default: return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT32;
    }
  }
  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return SPConvMMPluginConsts::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return SPConvMMPluginConsts::version; };
  void setPluginNamespace(AsciiChar const* pluginNamespace) NOEXCEPT override { mNamespace = pluginNamespace; };
  AsciiChar const* getPluginNamespace() const NOEXCEPT override { return mNamespace; };

  /**
   * Runtime Part
   * */
  void configurePlugin(const DynamicPluginTensorDesc* in,
                       int32_t nbInputs,
                       const DynamicPluginTensorDesc* out,
                       int32_t nbOutputs) NOEXCEPT override {
    size_t wNumEl = p.kernelVol * p.inChannels * p.outChannels;
    size_t bNumEl = p.outChannels;
    if (in[0].desc.type == DataType::kHALF) {
      std::vector<half> wbHalf(wb.size());
      for (int i = 0; i < wb.size(); i++) { wbHalf[i] = static_cast<half>(wb[i]); }
      cudaMemcpy(wRt->data(), &wbHalf[0], wNumEl * sizeof(wbHalf[0]), cudaMemcpyHostToDevice);
      if (p.withBias) cudaMemcpy(bRt->data(), &wbHalf[wNumEl], bNumEl * sizeof(wbHalf[0]), cudaMemcpyHostToDevice);
      return;
    }
    cudaMemcpy(wRt->data(), &wb[0], wNumEl * sizeof(wb[0]), cudaMemcpyHostToDevice);
    if (p.withBias) cudaMemcpy(bRt->data(), &wb[wNumEl], bNumEl * sizeof(wb[0]), cudaMemcpyHostToDevice);
  };
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
     *        0: inFeats              [float/half]    [mMaxNumActIn, inChannels]
     *        1: numActIn             [int32]         [1]
     *        2: numActOut            [int32]         [1]
     *        3: index                [int32]         [3, kVol * (mMaxNumActIn + 128)]
     *        4: numIndex             [int32]         [1]
     *    Output:
     *        0: outFeats             [float/half]    [mMaxNumActOut,
     * outChannels]
     * */
    auto* indexPtr = reinterpret_cast<const int32_t*>(inputs[3]);
    auto* numIndexPtr = reinterpret_cast<const int32_t*>(inputs[4]);
    int numIndicesMax = inputDesc[3].dims.d[1];
    auto gatherInPtr = indexPtr;
    auto scatterOutPtr = indexPtr + numIndicesMax;
    auto kernelOffsetPtr = indexPtr + 2 * numIndicesMax;

    if (p.inverse) {
      gatherInPtr = scatterOutPtr;
      scatterOutPtr = indexPtr;
    }

    Ref1D<const int32_t> gatherIn(gatherInPtr, {numIndicesMax});
    Ref1D<const int32_t> scatterOut(scatterOutPtr, {numIndicesMax});
    Ref1D<const int32_t> kernelOffset(kernelOffsetPtr, {numIndicesMax});

    auto gpu = utils::GPU(stream);
    if (inputDesc[0].type == DataType::kHALF) {
      using dtype = cutlass::half_t;
      Ref2D<dtype> outFeats = fromTensorRT<2, dtype>(outputs[0], outputDesc[0]);
      Ref2D<const dtype> inFeats = fromTensorRT<2, const dtype>(inputs[0], inputDesc[0]);
      Ref3D<const dtype> filters(reinterpret_cast<const dtype*>(wRt->data()),
                                 {p.kernelVol, p.inChannels, p.outChannels});
      Ref1D<const dtype> bias(p.withBias ? reinterpret_cast<dtype*>(bRt->data()) : nullptr, {p.outChannels});
      if (inFeats.size(1) % 8 == 0 && filters.size(1) % 8 == 0) {
        if (inFeats.size(1) <= 16) {
          spconv::func::indexedSpConv<MMOp1>(gpu, outFeats, inFeats, filters, bias, gatherIn, scatterOut, kernelOffset,
                                             numIndexPtr);
        } else {
          spconv::func::indexedSpConv<MMOp2>(gpu, outFeats, inFeats, filters, bias, gatherIn, scatterOut, kernelOffset,
                                             numIndexPtr);
        }
      } else
        spconv::func::indexedSpConv<MMOp3>(gpu, outFeats, inFeats, filters, bias, gatherIn, scatterOut, kernelOffset,
                                           numIndexPtr);
    }
    return 0;
  }
};

class SPConvMMPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  SPConvMMPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return SPConvMMPluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return SPConvMMPluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new SPConvMMPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};

REGISTER_TENSORRT_PLUGIN(SPConvMMPluginCreator);

}  // namespace spconv
