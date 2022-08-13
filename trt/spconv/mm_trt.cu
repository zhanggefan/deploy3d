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

struct SpConvMMPluginConsts {
  static constexpr const char* name = "SpConvMM";
  static constexpr const char* version = "1.0";
};

#pragma pack(push, 1)
struct SpConvMMPluginParam {
  int32_t kernelVol;
  int32_t inChannels;
  int32_t outChannels;
  int32_t maxNumActOut;
  bool subM;
  bool inverse;
  bool withBias;
};
#pragma pack(pop)

class SpConvMMPlugin : public IPluginV2DynamicExt {
 public:
 private:
  SpConvMMPluginParam p;
  std::vector<float> wb;
  std::shared_ptr<DeviceVector> wRt;
  std::shared_ptr<DeviceVector> bRt;
  std::vector<int32_t> numBufAndBufSegLen;
  cublasHandle_t cublas;
  DeviceVector cublasWS;
  const char* mNamespace;
  static constexpr size_t cublasWSSize = 16777216;

 public:
  /**
   * Lifecycle Part
   * */
  SpConvMMPlugin() = delete;
  SpConvMMPlugin(const SpConvMMPluginParam& param,
                 const std::vector<float>& weightAndBias,
                 const std::shared_ptr<DeviceVector>& wRuntime,
                 const std::shared_ptr<DeviceVector>& bRuntime)
      : p(param), wb(weightAndBias), wRt(wRuntime), bRt(bRuntime), numBufAndBufSegLen(param.kernelVol + 1),
        cublasWS(cublasWSSize) {
    cublasCreate_v2(&cublas);
    cublasSetWorkspace_v2(cublas, cublasWS.data(), cublasWSSize);
  }
  SpConvMMPlugin(const void* data, size_t length) : cublasWS(cublasWSSize) {
    SerializeStream s(data, length);
    s >> p;
    wb.resize((length - s.curPos()) / sizeof(float));
    s.loadRange(wb.begin(), wb.end());
    wRt = std::make_shared<DeviceVector>(p.kernelVol * p.inChannels * p.outChannels * sizeof(float));
    bRt = std::make_shared<DeviceVector>(p.outChannels * sizeof(float));
    numBufAndBufSegLen.resize(p.kernelVol + 1);
    cublasCreate_v2(&cublas);
    cublasSetWorkspace_v2(cublas, cublasWS.data(), cublasWSSize);
  }
  size_t getSerializationSize() const NOEXCEPT override { return sizeof(p) + wb.size() * sizeof(wb[0]); }
  void serialize(void* buffer) const NOEXCEPT override {
    SerializeStream s(buffer);
    s << p;
    s.dumpRange(wb.begin(), wb.end());
  }
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new SpConvMMPlugin(p, wb, wRt, bRt);
    obj->setPluginNamespace(mNamespace);
    return obj;
  }
  int32_t initialize() NOEXCEPT override { return 0; };
  void terminate() NOEXCEPT override{};
  void destroy() NOEXCEPT override {
    cublasDestroy_v2(cublas);
    delete this;
  };

  /**
   * IO Part:
   *    Input:
   *        0: inFeats              [float/half]    [mMaxNumActIn, inChannels]
   *        1: numActIn             [int32]         [1]
   *        2: numActOut            [int32]         [1]
   *        3: index                [int32]         [3, kVol * mMaxNumActIn]
   *        4: (numBuf, bufSegLen)  [int32]         [1 + kVol]
   *    Output:
   *        0: outFeats             [float/half]    [mMaxNumActOut,
   * outChannels]
   * */
  int32_t getNbOutputs() const NOEXCEPT override { return 1; };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    DimsExprs outFeats(inputs[0]);
    if (!p.subM) outFeats.d[0] = exprBuilder.constant(p.maxNumActOut);
    outFeats.d[1] = exprBuilder.constant(p.outChannels);
    return outFeats;
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
  AsciiChar const* getPluginType() const NOEXCEPT override { return SpConvMMPluginConsts::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return SpConvMMPluginConsts::version; };
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
    switch (inputs[0].type) {
    case DataType::kHALF: return inputs[0].dims.d[0] * p.kernelVol * (p.outChannels + p.inChannels) * sizeof(half);
    default: return inputs[0].dims.d[0] * p.kernelVol * (p.outChannels + p.inChannels) * sizeof(float);
    }
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
     *        3: index                [int32]         [3, kVol * mMaxNumActIn]
     *        4: (numBuf, bufSegLen)  [int32]         [1 + kVol]
     *    Output:
     *        0: outFeats             [float/half]    [mMaxNumActOut,
     * outChannels]
     * */
    int32_t numActIn, numActOut;
    cudaMemcpyAsync(&numActIn, inputs[1], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&numActOut, inputs[2], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(numBufAndBufSegLen.data(), inputs[4], numBufAndBufSegLen.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int32_t numBuf = numBufAndBufSegLen[0];
    Ref1D<int32_t> bufferKernelNumHost(&numBufAndBufSegLen[1], {p.kernelVol});
    auto* indexPtr = reinterpret_cast<int32_t*>(const_cast<void*>(inputs[3]));

    auto indexBufferFromInPtr = indexPtr;
    auto indexBufferToOutPtr = indexPtr + p.kernelVol * inputDesc[0].dims.d[0];
    auto indexBufferOffsetPtr = indexPtr + 2 * p.kernelVol * inputDesc[0].dims.d[0];

    if (p.inverse) {
      indexBufferFromInPtr = indexBufferToOutPtr;
      indexBufferToOutPtr = indexPtr;
      int tmp = numActIn;
      numActIn = numActOut;
      numActOut = tmp;
    }

    Ref1D<int32_t> bufferFromIn(indexBufferFromInPtr, {numBuf});
    Ref1D<int32_t> bufferToOut(indexBufferToOutPtr, {numBuf});
    Ref1D<int32_t> bufferOffset(indexBufferOffsetPtr, {numBuf});
    auto gpu = utils::GPU(stream, cublas);
    if (inputDesc[0].type == DataType::kHALF) {
      using dtype = half;
      Ref2D<dtype> bufMMIn(reinterpret_cast<dtype*>(workspace), {numActIn * p.kernelVol, p.inChannels});
      Ref2D<dtype> bufMMOut(reinterpret_cast<dtype*>(workspace) + bufMMIn.numel(),
                            {numActIn * p.kernelVol, p.outChannels});
      Ref2D<dtype> outFeats(reinterpret_cast<dtype*>(outputs[0]), {numActOut, p.outChannels});
      Ref2D<dtype> inFeats(reinterpret_cast<dtype*>(const_cast<void*>(inputs[0])), {numActIn, p.inChannels});
      Ref3D<dtype> filters(reinterpret_cast<dtype*>(wRt->data()), {p.kernelVol, p.inChannels, p.outChannels});
      dtype* biasPtr = nullptr;
      if (p.withBias) biasPtr = reinterpret_cast<dtype*>(bRt->data());
      Ref1D<dtype> bias(biasPtr, {p.outChannels});
      if (p.subM) {
        spconv::func::indexSubM<4, dtype, int32_t>(gpu, bufMMIn, bufMMOut, outFeats, inFeats, filters, bias,
                                                   bufferFromIn, bufferToOut, bufferOffset, bufferKernelNumHost);
      } else {
        spconv::func::indexConv<4, dtype, int32_t>(gpu, bufMMIn, bufMMOut, outFeats, inFeats, filters, bias,
                                                   bufferFromIn, bufferToOut, bufferOffset, bufferKernelNumHost);
      }
    } else {
      using dtype = float;
      Ref2D<dtype> bufMMIn(reinterpret_cast<dtype*>(workspace), {numActIn * p.kernelVol, p.inChannels});
      Ref2D<dtype> bufMMOut(reinterpret_cast<dtype*>(workspace) + bufMMIn.numel(),
                            {numActIn * p.kernelVol, p.outChannels});
      Ref2D<dtype> outFeats(reinterpret_cast<dtype*>(outputs[0]), {numActOut, p.outChannels});
      Ref2D<dtype> inFeats(reinterpret_cast<dtype*>(const_cast<void*>(inputs[0])), {numActIn, p.inChannels});
      Ref3D<dtype> filters(reinterpret_cast<dtype*>(wRt->data()), {p.kernelVol, p.inChannels, p.outChannels});
      dtype* biasPtr = nullptr;
      if (p.withBias) biasPtr = reinterpret_cast<dtype*>(bRt->data());
      Ref1D<dtype> bias(biasPtr, {p.outChannels});
      if (p.subM) {
        spconv::func::indexSubM<4, dtype, int32_t>(gpu, bufMMIn, bufMMOut, outFeats, inFeats, filters, bias,
                                                   bufferFromIn, bufferToOut, bufferOffset, bufferKernelNumHost);
      } else {
        spconv::func::indexConv<4, dtype, int32_t>(gpu, bufMMIn, bufMMOut, outFeats, inFeats, filters, bias,
                                                   bufferFromIn, bufferToOut, bufferOffset, bufferKernelNumHost);
      }
    }
    return 0;
  }
};

class SpConvMMPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  SpConvMMPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return SpConvMMPluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return SpConvMMPluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new SpConvMMPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};

REGISTER_TENSORRT_PLUGIN(SpConvMMPluginCreator);

}  // namespace spconv
