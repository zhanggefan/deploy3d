from symfun.ops.decoder import YoloX3dDecode
from symfun.trt_utils import TRTPluginModule
import torch


def _decode(preds_dtype, bev_shape, topk, class_ids):
    voxel_config = torch.tensor([-76.8, -76.8, -2, 0.1, 0.1, 0.15])
    cls_scores = torch.randn(2, len(class_ids), *bev_shape, dtype=preds_dtype)
    bbox_preds = torch.randn(2, 9, *bev_shape, dtype=preds_dtype)
    output = TRTPluginModule.forward(
        YoloX3dDecode,
        input_tensors=(cls_scores, bbox_preds, voxel_config,),
        configs=[8, class_ids, topk])
    assert len(output) == 3 * len(class_ids)
    for group_idx, cls_idx in enumerate(class_ids):
        assert (output[3 * group_idx] == cls_idx).all()


def test_decode_0():
    _decode(torch.float32, [192, 192], 256, [1, 2])


def test_decode_1():
    _decode(torch.float32, [10, 10], 256, [1, 2])


def test_decode_2():
    _decode(torch.float16, [192, 192], 256, [1, 2])


def test_decode_3():
    _decode(torch.float16, [10, 10], 256, [1, 2])
