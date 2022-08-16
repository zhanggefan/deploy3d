#include "common/launch.cuh"
#include "common/macros.h"
#include "common/refnd.h"
#include "cuda_fp16.hpp"
#include "iou_rotated.cuh"
#include <NvInfer.h>
#include <cstring>
#include <vector>

#define NOEXCEPT noexcept

namespace decoder {
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
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

namespace kernel {
template <typename T1, typename T2> HOST_DEVICE_INLINE ssize_t DivUp(const T1 a, const T2 b) {
  return (a + b - 1) / b;
}

DEVICE_INLINE void box3d_to_bev(const float* box3d, iou_rotated::RotatedBox<float>& bev) {
  bev.x_ctr = box3d[0];
  bev.y_ctr = box3d[1];
  bev.w = box3d[3];
  bev.h = box3d[4];
  bev.a = box3d[6];
}

DEVICE_INLINE float iou3d(const float* box_a, const float* box_b) {
  // params: box_a (7) [cx, cy, cz, dx, dy, dz, angle]
  // params: box_b (7) [cx, cy, cz, dx, dy, dz, angle]
  iou_rotated::RotatedBox<float> bev_a, bev_b;
  box3d_to_bev(box_a, bev_a);
  box3d_to_bev(box_b, bev_b);
  //  float va = box_a[3] * box_a[4] * box_a[5];
  //  float vb = box_b[3] * box_b[4] * box_b[5];
  //  float zbottom = max(box_a[2], box_b[2]);
  //  float ztop = min(box_a[2] + box_a[5], box_b[2] + box_b[5]);
  //  float h_overlap = ztop > zbottom ? ztop - zbottom : 0;
  //  float v_overlap = h_overlap * bev_overlap(bev_a, bev_b);
  float va = box_a[3] * box_a[4];
  float vb = box_b[3] * box_b[4];
  float v_overlap = iou_rotated::rotated_boxes_intersection(bev_a, bev_b);
  return v_overlap / fmaxf(va + vb - v_overlap, iou_rotated::EPS);
}

__global__ void
overlap_matrix_kernel(const float* boxes, uint64_t* overlapMatrix, const int nbBox, const float nmsOverlapThreshold) {
  // params: boxes (N, 7) [cx, cy, cz, dx, dy, dz, angle]
  // params: overlapMatrix (N, N/THREADS_PER_BLOCK_NMS)
  const int nbOverlapHashBlocks = DivUp(nbBox, THREADS_PER_BLOCK_NMS);
  const int b = blockIdx.x;
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.z;

  if (row_start > col_start) return;

  const float* boxesOffset = boxes + b * nbBox * 7;
  uint64_t* overlapMatrixOffset = overlapMatrix + b * nbBox * nbOverlapHashBlocks;

  const int cur_row = row_start * THREADS_PER_BLOCK_NMS + threadIdx.x;
  const int cur_col = col_start * THREADS_PER_BLOCK_NMS + threadIdx.x;

  const int col_size = fminf(nbBox - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

  __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

  if (cur_col < nbBox) {
    const float* from = boxesOffset + cur_col * 7;
    float* to = block_boxes + threadIdx.x * 7;
    to[0] = from[0];
    to[1] = from[1];
    to[2] = from[2];
    to[3] = from[3];
    to[4] = from[4];
    to[5] = from[5];
    to[6] = from[6];
  }
  __syncthreads();

  if (cur_row < nbBox) {
    const float* cur_box = boxesOffset + cur_row * 7;
    uint64_t t = 0;
    int i = (row_start == col_start) ? (threadIdx.x + 1) : 0;
    for (; i < col_size; i++) {
      if (iou3d(cur_box, block_boxes + i * 7) >= nmsOverlapThreshold) { t |= 1ULL << i; }
    }
    overlapMatrixOffset[cur_row * nbOverlapHashBlocks + col_start] = t;
  }
}

__global__ void score_nms_kernel(const uint64_t* overlapMatrix, float* score, const int nbBox) {
  const int nbOverlapHashBlocks = DivUp(nbBox, THREADS_PER_BLOCK_NMS);
  const int b = blockIdx.x;

  const uint64_t* overlapMatrixOffset = overlapMatrix + b * nbBox * nbOverlapHashBlocks;
  float* scoreOffset = score + b * nbBox;

  __shared__ extern uint64_t overlapHash[];  // len: nbOverlapHashBlocks
  for (int i = 0; i < nbOverlapHashBlocks; i++) overlapHash[i] = 0;

  for (int i = 0; i < nbBox; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(overlapHash[nblock] & (1ULL << inblock))) {
      const uint64_t* p = overlapMatrixOffset + i * nbOverlapHashBlocks;
      for (int j = nblock; j < nbOverlapHashBlocks; j++) overlapHash[j] |= p[j];
    } else {
      scoreOffset[i] = -1.0f;
    }
  }
}
}  // namespace kernel

struct NMS3dPluginConsts {
  static constexpr const char* name = "NMS3d";
  static constexpr const char* version = "2.0";
};

class NMS3dPlugin : public IPluginV2DynamicExt {
 private:
  const char* mNamespace;
  const float mNmsOverlapThreshold = 0.1f;

 public:
  NMS3dPlugin() = delete;
  NMS3dPlugin(const void* data, size_t length){};
  size_t getSerializationSize() const NOEXCEPT override { return 0; }
  void serialize(void* buffer) const NOEXCEPT override{};
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new NMS3dPlugin(nullptr, 0);
    obj->setPluginNamespace(mNamespace);
    return obj;
  }
  int32_t initialize() NOEXCEPT override { return 0; };
  void terminate() NOEXCEPT override{};
  void destroy() NOEXCEPT override { delete this; };

  /**
   * IO Part:
   *    Input:
   *        0: inScores             [float]     [b, topK]
   *        1: boxes                [float]     [b, topK, 7]
   *    Output:
   *        0: outScores            [float]     [b, topK]
   * */
  int32_t getNbOutputs() const NOEXCEPT override { return 1; };
  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    return inputs[0];
  }
  DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const NOEXCEPT override {
    return DataType::kFLOAT;
  }
  bool supportsFormatCombination(int32_t pos,
                                 const PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) NOEXCEPT override {
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
  }
  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return NMS3dPluginConsts::name; }
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return NMS3dPluginConsts::version; };
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
    const int nbBatch = inputs[0].dims.d[0], nbBox = inputs[0].dims.d[1];
    return nbBatch * nbBox * kernel::DivUp(nbBox, THREADS_PER_BLOCK_NMS) * sizeof(uint64_t);
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
     *        0: inScores             [float]     [b, topK]
     *        1: boxes                [float]     [b, topK, 7]
     *    Output:
     *        0: outScores            [float]     [b, topK]
     * */
    if (inputDesc[0].type != DataType::kFLOAT || inputDesc[0].type != DataType::kFLOAT)
      throw std::runtime_error("Input type is not supported!");

    const Dims& boxesDims = inputDesc[0].dims;
    const int nbBatch = boxesDims.d[0], nbBox = boxesDims.d[1];
    const int nbOverlapHashBlocks = kernel::DivUp(nbBox, THREADS_PER_BLOCK_NMS);

    cudaMemcpyAsync(outputs[0], inputs[0], nbBatch * nbBox * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    float* nmsScores = reinterpret_cast<float*>(outputs[0]);

    const float* boxes = reinterpret_cast<const float*>(inputs[1]);

    uint64_t* overlapMatrix = reinterpret_cast<uint64_t*>(workspace);
    cudaMemsetAsync(workspace, 0, nbBatch * nbBox * nbOverlapHashBlocks * sizeof(uint64_t), stream);

    dim3 blocks(nbBatch, nbOverlapHashBlocks, nbOverlapHashBlocks);
    dim3 threads(THREADS_PER_BLOCK_NMS);
    kernel::overlap_matrix_kernel<<<blocks, threads, 0, stream>>>(boxes, overlapMatrix, nbBox, mNmsOverlapThreshold);
    kernel::score_nms_kernel<<<nbBatch, 1, nbOverlapHashBlocks * sizeof(uint64_t), stream>>>(overlapMatrix, nmsScores,
                                                                                             nbBox);
    return 0;
  };
};

class NMS3dPluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  NMS3dPluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return NMS3dPluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return NMS3dPluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new NMS3dPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;  // the only way when creating TRT_PluginV2
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};
REGISTER_TENSORRT_PLUGIN(NMS3dPluginCreator);
}  // namespace decoder