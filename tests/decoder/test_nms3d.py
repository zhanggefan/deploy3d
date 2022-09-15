from deploy3d.symfun.ops.decoder import NMS3d
from deploy3d.symfun.trt_utils import TRTPluginModule
import numpy as np
import torch
from mmcv.ops.nms import nms_rotated


def _nms3d(batch, topk):
    in_scores = 1 - torch.arange(topk, dtype=torch.float32) / topk
    in_scores = in_scores[None, :].repeat(batch, 1)
    xyz = torch.randn([batch, topk, 3], dtype=torch.float32)
    dxyz = torch.rand([batch, topk, 3], dtype=torch.float32) + 0.5
    ang = (torch.rand([batch, topk, 1], dtype=torch.float32) - 0.5) * 2 * np.pi
    bbox_preds = torch.cat([xyz, dxyz, ang], dim=-1)
    nmsed_score = TRTPluginModule.forward(
        NMS3d,
        input_tensors=(in_scores, bbox_preds))
    keep_mask = nmsed_score >= 0
    bevbox = torch.cat([xyz[..., :2], dxyz[..., :2], ang], dim=-1)
    for i in range(batch):
        _, keep_idx = nms_rotated(bevbox[i], in_scores[i], 0.1)
        gt_keep_mask = torch.zeros_like(keep_mask[i])
        gt_keep_mask[keep_idx] = True
        assert (gt_keep_mask == keep_mask[i]).all()


def test_nms3d_0():
    _nms3d(1, 256)


def test_nms3d_1():
    _nms3d(2, 256)


def test_nms3d_2():
    _nms3d(1, 128)


def test_nms3d_3():
    _nms3d(2, 128)
