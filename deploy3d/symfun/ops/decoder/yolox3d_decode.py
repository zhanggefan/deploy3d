import torch
import numpy as np


class YoloX3dDecode(torch.autograd.Function):
    """
    /** input:
     *    cls_scores:       float32/16 [b, numClassId, y, x]
     *    bbox_preds:       float32/16 [b, 9 + a, y, x]
     *    voxel_config:     float32 [6]
     *  output:
     *    -- repeat --
     *      topk_class_ids: int32      [b, topK]
     *      topk_scores:    float32    [b, topK]
     *      topk_boxes:     float32    [b, topK, 7 + a]
     *    -- repeat end --
     * */
    """

    @staticmethod
    def forward(ctx, cls_scores: torch.Tensor,
                bbox_preds: torch.Tensor,
                voxel_config: torch.Tensor,
                stride, class_ids, topk):
        num_class_ids = len(class_ids)
        batch_size = cls_scores.shape[0]
        ret = []
        topk = min(topk, int(cls_scores.shape[2]) * int(cls_scores.shape[3]))
        a = int(bbox_preds.shape[1]) - 9
        for _ in range(num_class_ids):
            topk_class_ids = cls_scores.new_zeros((batch_size, topk),
                                                  dtype=torch.int32)
            topk_scores = cls_scores.new_zeros((batch_size, topk))
            topk_boxes = cls_scores.new_zeros((batch_size, topk, 7 + a))
            ret += [topk_class_ids, topk_scores, topk_boxes]
        return tuple(ret)

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 cls_scores: torch._C.Value,
                 bbox_preds: torch._C.Value,
                 voxel_config: torch._C.Value,
                 stride, class_ids, topk):
        data = np.array(stride, dtype=np.int32).tobytes()
        data += np.array(topk, dtype=np.int32).tobytes()
        num_class_ids = len(class_ids)
        assert num_class_ids <= 8
        class_ids = tuple(class_ids) + (-1,) * (8 - num_class_ids)
        data += np.array(class_ids, dtype=np.int32).tobytes()
        return g.op('TRT_PluginV2', cls_scores, bbox_preds, voxel_config,
                    name_s=b'YoloX3dDecode', data_s=data, namespace_s=b'',
                    version_s=b'2.0', outputs=3 * num_class_ids)


yolox3d_decode = YoloX3dDecode.apply
