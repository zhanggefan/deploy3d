from symfun.ops.point_voxel_mapping import GatherBack
from symfun.trt_utils import TRTPluginModule
import torch


def _gather_from(num_max_voxel, num_feats, num_act_voxel, num_pts, has_invalid_pts):
    reduced_feats = torch.randn((num_max_voxel, num_feats), dtype=torch.float32)
    scatter_to_base = torch.arange(-1 if has_invalid_pts else 0, num_act_voxel, dtype=torch.int32)
    assert num_pts - scatter_to_base.shape[0] >= 0
    scatter_to_extra = torch.randint(low=-1 if has_invalid_pts else 0, high=num_act_voxel,
                                     size=(num_pts - scatter_to_base.shape[0],), dtype=torch.int32)
    scatter_to = torch.cat([scatter_to_base, scatter_to_extra])
    scatter_to = scatter_to[torch.randperm(scatter_to.shape[0])]

    batch_point_feats = TRTPluginModule.forward(
        GatherBack,
        input_tensors=(reduced_feats,
                       scatter_to))
    batch_point_feats_tgt = reduced_feats[scatter_to[scatter_to.ge(0)].long()]
    assert torch.allclose(batch_point_feats_tgt, batch_point_feats[scatter_to.ge(0)].cpu(), atol=1e-3, rtol=1e-3)


def test_gather_from_1():
    _gather_from(num_max_voxel=65536, num_feats=9, num_act_voxel=65536, num_pts=200000, has_invalid_pts=True)


def test_gather_from_2():
    _gather_from(num_max_voxel=65536, num_feats=9, num_act_voxel=32768, num_pts=200000, has_invalid_pts=True)


def test_gather_from_3():
    _gather_from(num_max_voxel=65536, num_feats=9, num_act_voxel=65536, num_pts=200000, has_invalid_pts=False)


def test_gather_from_4():
    _gather_from(num_max_voxel=65536, num_feats=9, num_act_voxel=32768, num_pts=200000, has_invalid_pts=False)
