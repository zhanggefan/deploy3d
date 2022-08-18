#include "common/cub.cuh"
#include "common/launch.cuh"
#include "common/refnd.h"
#include "cuda_fp16.hpp"

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
using utils::nd::Ref4D;
using utils::nd::Size;
using utils::nd::Vec;
using namespace nvinfer1;

namespace kernel {

template <typename T, typename classIdx_T>
__global__ void decode(Ref2D<int32_t> topk_class_ids,
                       Ref2D<float> topk_scores,
                       Ref3D<float> topk_boxes,
                       const int32_t class_i,
                       const int32_t stride,
                       const classIdx_T class_id,
                       const Ref3D<int32_t> sorted_indices,
                       const Ref3D<T> sorted_scores,
                       const Ref4D<const T> bbox_preds,
                       const Ref1D<const float> pillar_config) {
  const int32_t cls = class_id[class_i];
  const int32_t topK = topk_class_ids.size(1);
  const float x_min = pillar_config[0], y_min = pillar_config[1], z_min = pillar_config[2];
  const float feat_stride = static_cast<float>(stride) * pillar_config[3];

  auto num_output = topk_class_ids.numel();
  for (auto idx : utils::launch::KernelLoopX(num_output)) {
    int32_t b = idx / topK;
    int32_t rank = idx % topK;
    int32_t pos = sorted_indices(b, class_i, rank);

    auto x_base = static_cast<float>(pos % bbox_preds.size(3));
    auto y_base = static_cast<float>(pos / bbox_preds.size(3));

    const T* pred_box = &bbox_preds(b, 0, 0, 0) + pos;
    float score, cx, cy, cz, dx, dy, dz, yaw, dir_sine, dir_cosine;
    auto box_param_stride = bbox_preds.stride(1);

    score = sorted_scores(b, class_i, rank);

    cx = float(*pred_box) + x_base;
    bool cx_valid = (cx <= bbox_preds.size(3) && cx >= 0);
    cx = x_min + feat_stride * cx;
    pred_box += box_param_stride;
    cy = float(*pred_box) + y_base;
    bool cy_valid = (cy <= bbox_preds.size(2) && cy >= 0);
    cy = y_min + feat_stride * cy;
    pred_box += box_param_stride;
    cz = float(*pred_box);
    pred_box += box_param_stride;
    dx = feat_stride * exp(float(*pred_box));
    pred_box += box_param_stride;
    dy = feat_stride * exp(float(*pred_box));
    pred_box += box_param_stride;
    dz = exp(float(*pred_box));
    pred_box += box_param_stride;
    yaw = float(*pred_box);
    pred_box += box_param_stride;
    dir_sine = float(*pred_box);
    pred_box += box_param_stride;
    dir_cosine = float(*pred_box);

    float dir = atan2(dir_sine, dir_cosine);
    bool yaw_valid = ((dir_sine != 0) || (dir_cosine != 0));

    if (!(isfinite(cx) && isfinite(cy) && isfinite(cz) && isfinite(dx) && isfinite(dy) && isfinite(dz) &&
          isfinite(yaw) && dx > 0 && dy > 0 && dz > 0 && yaw_valid && cx_valid && cy_valid)) {
      score = -1.0f;
      cx = 10000.0f;
      cy = 10000.0f;
      cz = 10000.0f;
      dx = 1.f;
      dy = 1.f;
      dz = 1.f;
      yaw = 0.f;
    } else {
      if (0.9f * dx < dy && dy < 1.11f * dx) {
        // small hw ratio. use dir instead of yaw
        yaw = dir;
        float t = max(dx, dy);
        dx = t;
        dy = t;
      } else {
        // large hw ratio. use yaw corrected by dir
        if (dy > dx) {
          yaw += M_PI_2;
          float t = dy;
          dy = dx;
          dx = t;
        }
        yaw += floorf((dir - yaw) / M_PI + 0.5) * M_PI;
        yaw -= floorf(yaw / (2 * M_PI) + 0.5) * (2 * M_PI);
      }
    }
    topk_class_ids[idx] = cls;
    topk_scores[idx] = score;
    auto topk_boxes_offset = &topk_boxes[idx * 7];
    topk_boxes_offset[0] = cx;
    topk_boxes_offset[1] = cy;
    topk_boxes_offset[2] = cz;
    topk_boxes_offset[3] = dx;
    topk_boxes_offset[4] = dy;
    topk_boxes_offset[5] = dz;
    topk_boxes_offset[6] = yaw;
  }
}

__global__ void initIndex(Ref3D<int32_t> index) {
  auto num_input = index.numel();
  for (auto idx : utils::launch::KernelLoopX(num_input)) { index[idx] = idx % index.size(2); }
}
}  // namespace kernel

struct YoloX3dDecodePluginConsts {
  static constexpr const char* name = "YoloX3dDecode";
  static constexpr const char* version = "2.0";
};

#pragma pack(push, 1)
struct YoloX3dDecodePluginParam {
  int32_t stride;
  int32_t topk;
  Vec<8, int32_t> classIds;
};
#pragma pack(pop)

class YoloX3dDecodePlugin : public IPluginV2DynamicExt {
 private:
  YoloX3dDecodePluginParam p;
  std::unique_ptr<int32_t[]> mTopKSegmentHostPtr;
  std::unique_ptr<utils::mem::DeviceVector> mTopKSegmentDevicePtr;
  const char* mNamespace;

 public:
  YoloX3dDecodePlugin() = delete;
  YoloX3dDecodePlugin(const YoloX3dDecodePluginParam& param) : p(param){};
  YoloX3dDecodePlugin(const void* data, size_t length) {
    SerializeStream s(data, length);
    s >> p;
  }
  size_t getSerializationSize() const NOEXCEPT override { return sizeof(p); }
  void serialize(void* buffer) const NOEXCEPT override {
    SerializeStream s(buffer);
    s << p;
  }
  IPluginV2DynamicExt* clone() const NOEXCEPT override {
    auto* obj = new YoloX3dDecodePlugin(p);
    obj->setPluginNamespace(mNamespace);
    return obj;
  }
  int initialize() NOEXCEPT override { return 0; };
  void terminate() NOEXCEPT override{};
  void destroy() NOEXCEPT override { delete this; };

  /** input:
   *    cls_scores:       float32/16 [b, numClassId, y, x]
   *    bbox_preds:       float32/16 [b, 9, y, x]
   *    voxel_config:     float32 [6]
   *  output:
   *    -- repeat --
   *      topk_class_ids: int32      [b, topK]
   *      topk_scores:    float32    [b, topK]
   *      topk_boxes:     float32    [b, topK, 7]
   *    -- repeat end --
   * */
  int getNbOutputs() const NOEXCEPT override {
    for (int i = 0; i < p.classIds.size(); i++)
      if (p.classIds[i] < 0) return 3 * i;
    return 3 * p.classIds.size();
  }
  DimsExprs getOutputDimensions(int outputIndex,
                                const DimsExprs* inputs,
                                int nbInputs,
                                IExprBuilder& exprBuilder) NOEXCEPT override {
    auto size_bev = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
    auto size_topk = exprBuilder.constant(p.topk);
    auto size_topk_clamped = exprBuilder.operation(DimensionOperation::kMIN, *size_bev, *size_topk);
    switch (outputIndex % 3) {
    case 0:
    case 1: return {2, {inputs[0].d[0], size_topk_clamped}};
    default: return {3, {inputs[0].d[0], size_topk_clamped, exprBuilder.constant(7)}};
    }
  }
  DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const NOEXCEPT override {
    switch (index % 3) {
    case 0: return DataType::kINT32;
    case 1:
    default: return DataType::kFLOAT;
    }
  }
  bool
  supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) NOEXCEPT override {
    switch (pos) {
    case 0:
      return inOut[pos].format == TensorFormat::kLINEAR &&
          (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
    case 1: return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == inOut[0].type;
    case 2: return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
    default: break;
    }
    switch ((pos - 3) % 3) {
    case 0: return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT32;
    case 1:
    default: return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
    }
  }

  /**
   * Utility Part
   * */
  AsciiChar const* getPluginType() const NOEXCEPT override { return YoloX3dDecodePluginConsts::name; };
  AsciiChar const* getPluginVersion() const NOEXCEPT override { return YoloX3dDecodePluginConsts::version; };
  void setPluginNamespace(AsciiChar const* pluginNamespace) NOEXCEPT override { mNamespace = pluginNamespace; };
  AsciiChar const* getPluginNamespace() const NOEXCEPT override { return mNamespace; };

  /**
   * Runtime Part
   * */
  void configurePlugin(const DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const DynamicPluginTensorDesc* out,
                       int nbOutputs) NOEXCEPT {
    int32_t max_size_batch = in[0].max.d[0];
    int32_t max_size_classid = in[0].max.d[1];
    int32_t sort_num_segments = max_size_batch * max_size_classid;
    mTopKSegmentHostPtr.reset(new int[sort_num_segments + 1]);
    mTopKSegmentDevicePtr.reset(new utils::mem::DeviceVector((sort_num_segments + 1) * sizeof(int32_t)));
  }
  size_t getWorkspaceSize(const PluginTensorDesc* inputs,
                          int nbInputs,
                          const PluginTensorDesc* outputs,
                          int nbOutputs) const NOEXCEPT {
    auto cls_scores_shape = fromTensorRT<4>(inputs[0]);

    int32_t size_batch = cls_scores_shape.size(0);
    int32_t size_classid = cls_scores_shape.size(1);
    int32_t size_bev = cls_scores_shape.stride(1);
    int32_t sort_num_items = cls_scores_shape.numel();
    int32_t sort_num_segments = size_batch * size_classid;

    for (int i = 0; i < sort_num_segments + 1; i++) mTopKSegmentHostPtr[i] = i * size_bev;
    cudaMemcpy(mTopKSegmentDevicePtr->data(), mTopKSegmentHostPtr.get(), (sort_num_segments + 1) * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    auto seg_begin = reinterpret_cast<int32_t*>(mTopKSegmentDevicePtr->data());
    auto seg_end = seg_begin + 1;

    if (inputs[0].type == DataType::kFLOAT) {
      using value_type = float;

      size_t workspaceBytes;
      CUB_NS_QUALIFIER::cub::DeviceSegmentedRadixSort::SortPairsDescending(
          nullptr, workspaceBytes, static_cast<value_type*>(nullptr), static_cast<value_type*>(nullptr),
          static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr), sort_num_items, sort_num_segments, seg_begin,
          seg_end);
      workspaceBytes += sort_num_items * sizeof(value_type);
      workspaceBytes += 2 * sort_num_items * sizeof(int32_t);
      return workspaceBytes;
    } else {
      using value_type = half;

      size_t workspaceBytes;
      CUB_NS_QUALIFIER::cub::DeviceSegmentedRadixSort::SortPairsDescending(
          nullptr, workspaceBytes, static_cast<value_type*>(nullptr), static_cast<value_type*>(nullptr),
          static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr), sort_num_items, sort_num_segments, seg_begin,
          seg_end);
      workspaceBytes += sort_num_items * sizeof(value_type);
      workspaceBytes += 2 * sort_num_items * sizeof(int32_t);
      return workspaceBytes;
    }
  }
  int enqueue(const PluginTensorDesc* inputDesc,
              const PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) NOEXCEPT {
    /** input:
     *    cls_scores:       float32/16 [b, numClassId, y, x]
     *    bbox_preds:       float32/16 [b, 9, y, x]
     *    voxel_config:     float32 [6]
     *  output:
     *    -- repeat --
     *      topk_class_ids: int32      [b, topK]
     *      topk_scores:    float32    [b, topK]
     *      topk_boxes:     float32    [b, topK, 7]
     *    -- repeat end --
     * */
    CHECK_CUDA_ERR();
    auto cls_scores_shape = fromTensorRT<4>(inputDesc[0]);

    int32_t size_batch = cls_scores_shape.size(0);
    int32_t size_classid = cls_scores_shape.size(1);
    int32_t size_bev = cls_scores_shape.stride(1);
    int32_t sort_num_items = cls_scores_shape.numel();
    int32_t sort_num_segments = size_batch * size_classid;

    for (int i = 0; i < sort_num_segments + 1; i++) mTopKSegmentHostPtr[i] = i * size_bev;
    cudaMemcpyAsync(mTopKSegmentDevicePtr->data(), mTopKSegmentHostPtr.get(), (sort_num_segments + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    auto seg_begin = reinterpret_cast<int32_t*>(mTopKSegmentDevicePtr->data());
    auto seg_end = seg_begin + 1;

    auto workspace_ptr = reinterpret_cast<uint8_t*>(workspace);

    cudaError_t err;
    if (inputDesc[0].type == DataType::kFLOAT) {
      using value_type = float;
      auto cls_scores = fromTensorRT<4, const value_type>(inputs[0], inputDesc[0]);
      auto bbox_preds = fromTensorRT<4, const value_type>(inputs[1], inputDesc[1]);
      auto pillar_config = fromTensorRT<1, const float>(inputs[2], inputDesc[2]);

      auto sorted_scores =
          Ref3D<value_type>(reinterpret_cast<value_type*>(workspace_ptr), {size_batch, size_classid, size_bev});
      workspace_ptr += sorted_scores.numby();
      auto indices = Ref3D<int32_t>(reinterpret_cast<int32_t*>(workspace_ptr), {size_batch, size_classid, size_bev});
      workspace_ptr += indices.numby();
      auto sorted_indices =
          Ref3D<int32_t>(reinterpret_cast<int32_t*>(workspace_ptr), {size_batch, size_classid, size_bev});
      auto temp = reinterpret_cast<void*>(workspace_ptr + sorted_indices.numby());

      kernel::initIndex<<<utils::launch::getBlocks(sort_num_items), utils::launch::CUDA_NUM_THREADS, 0, stream>>>(
          indices);

      size_t cubTempStorageBytes;
      err = CUB_NS_QUALIFIER::cub::DeviceSegmentedRadixSort::SortPairsDescending(
          nullptr, cubTempStorageBytes, static_cast<const value_type*>(nullptr), static_cast<value_type*>(nullptr),
          static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr), sort_num_items, sort_num_segments,
          seg_begin, seg_end, 0, sizeof(value_type) * 8, stream);
      CHECK_RETURN_STATUS(err);
      err = CUB_NS_QUALIFIER::cub::DeviceSegmentedRadixSort::SortPairsDescending(
          temp, cubTempStorageBytes, cls_scores.data(), sorted_scores.data(), indices.data(), sorted_indices.data(),
          sort_num_items, sort_num_segments, seg_begin, seg_end, 0, sizeof(value_type) * 8, stream);
      CHECK_RETURN_STATUS(err);

      for (int32_t class_i = 0; class_i < size_classid; class_i++) {
        auto topk_class_ids = fromTensorRT<2, int32_t>(outputs[3 * class_i + 0], outputDesc[3 * class_i + 0]);
        auto topk_scores = fromTensorRT<2, float>(outputs[3 * class_i + 1], outputDesc[3 * class_i + 1]);
        auto topk_boxes = fromTensorRT<3, float>(outputs[3 * class_i + 2], outputDesc[3 * class_i + 2]);

        int32_t num_output = topk_class_ids.numel();
        kernel::decode<value_type>
            <<<utils::launch::getBlocks(num_output), utils::launch::CUDA_NUM_THREADS, 0, stream>>>(
                topk_class_ids, topk_scores, topk_boxes, class_i, p.stride, p.classIds, sorted_indices, sorted_scores,
                bbox_preds, pillar_config);
      }
    } else {
      using value_type = half;
      auto cls_scores = fromTensorRT<4, const value_type>(inputs[0], inputDesc[0]);
      auto bbox_preds = fromTensorRT<4, const value_type>(inputs[1], inputDesc[1]);
      auto pillar_config = fromTensorRT<1, const float>(inputs[2], inputDesc[2]);

      auto sorted_scores =
          Ref3D<value_type>(reinterpret_cast<value_type*>(workspace_ptr), {size_batch, size_classid, size_bev});
      workspace_ptr += sorted_scores.numby();
      auto indices = Ref3D<int32_t>(reinterpret_cast<int32_t*>(workspace_ptr), {size_batch, size_classid, size_bev});
      workspace_ptr += indices.numby();
      auto sorted_indices =
          Ref3D<int32_t>(reinterpret_cast<int32_t*>(workspace_ptr), {size_batch, size_classid, size_bev});
      auto temp = reinterpret_cast<void*>(workspace_ptr + sorted_indices.numby());

      kernel::initIndex<<<utils::launch::getBlocks(sort_num_items), utils::launch::CUDA_NUM_THREADS, 0, stream>>>(
          indices);

      size_t cubTempStorageBytes;
      err = CUB_NS_QUALIFIER::cub::DeviceSegmentedRadixSort::SortPairsDescending(
          nullptr, cubTempStorageBytes, static_cast<const value_type*>(nullptr), static_cast<value_type*>(nullptr),
          static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr), sort_num_items, sort_num_segments,
          seg_begin, seg_end, 0, sizeof(value_type) * 8, stream);
      CHECK_RETURN_STATUS(err);
      err = CUB_NS_QUALIFIER::cub::DeviceSegmentedRadixSort::SortPairsDescending(
          temp, cubTempStorageBytes, cls_scores.data(), sorted_scores.data(), indices.data(), sorted_indices.data(),
          sort_num_items, sort_num_segments, seg_begin, seg_end, 0, sizeof(value_type) * 8, stream);
      CHECK_RETURN_STATUS(err);

      for (int32_t class_i = 0; class_i < size_classid; class_i++) {
        auto topk_class_ids = fromTensorRT<2, int32_t>(outputs[3 * class_i + 0], outputDesc[3 * class_i + 0]);
        auto topk_scores = fromTensorRT<2, float>(outputs[3 * class_i + 1], outputDesc[3 * class_i + 1]);
        auto topk_boxes = fromTensorRT<3, float>(outputs[3 * class_i + 2], outputDesc[3 * class_i + 2]);

        int32_t num_output = topk_class_ids.numel();
        kernel::decode<value_type>
            <<<utils::launch::getBlocks(num_output), utils::launch::CUDA_NUM_THREADS, 0, stream>>>(
                topk_class_ids, topk_scores, topk_boxes, class_i, p.stride, p.classIds, sorted_indices, sorted_scores,
                bbox_preds, pillar_config);
      }
    }
    return 0;
  }
};

class YoloX3dDecodePluginCreator : public IPluginCreator {
 private:
  std::string mNamespace;

 public:
  YoloX3dDecodePluginCreator() : mNamespace(""){};

  const char* getPluginName() const NOEXCEPT override { return YoloX3dDecodePluginConsts::name; };

  const char* getPluginVersion() const NOEXCEPT override { return YoloX3dDecodePluginConsts::version; };

  const PluginFieldCollection* getFieldNames() NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override {
    return nullptr;  // should not be called when creating TRT_PluginV2
  };

  IPluginV2DynamicExt*
  deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override {
    auto* obj = new YoloX3dDecodePlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
  };

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override { mNamespace = std::string(libNamespace); }

  const char* getPluginNamespace() const NOEXCEPT override { return mNamespace.c_str(); }
};
REGISTER_TENSORRT_PLUGIN(YoloX3dDecodePluginCreator);
}  // namespace decoder
