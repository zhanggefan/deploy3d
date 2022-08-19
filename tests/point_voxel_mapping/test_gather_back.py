from symfun.ops.point_voxel_mapping import GatherBack
from symfun.trt_utils import TRTPluginModule
import torch


def _gather_from(num_max_voxel, num_feats, num_act_voxel, num_pts, has_invalid_pts, fill_val=None):
    reduced_feats = torch.randn(
        (num_max_voxel,) + tuple(num_feats), dtype=torch.float32)
    scatter_to_base = torch.arange(-1 if has_invalid_pts else 0,
                                   num_act_voxel, dtype=torch.int32)
    assert num_pts - scatter_to_base.shape[0] >= 0
    scatter_to_extra = torch.randint(low=-1 if has_invalid_pts else 0, high=num_act_voxel,
                                     size=(num_pts - scatter_to_base.shape[0],), dtype=torch.int32)
    scatter_to = torch.cat([scatter_to_base, scatter_to_extra])
    scatter_to = scatter_to[torch.randperm(scatter_to.shape[0])]

    batch_point_feats = TRTPluginModule.forward(
        GatherBack,
        input_tensors=(reduced_feats,
                       scatter_to),
        configs=([fill_val] if fill_val is not None else []))
    valid_mask = scatter_to.ge(0)
    invalid_mask = scatter_to.lt(0)
    batch_point_feats_tgt = torch.randn(
        (num_pts,) + tuple(num_feats), dtype=torch.float32)
    batch_point_feats_tgt[valid_mask] = reduced_feats[scatter_to[valid_mask].long()]
    assert torch.allclose(batch_point_feats_tgt[valid_mask],
                          batch_point_feats[valid_mask].cpu(), atol=1e-3, rtol=1e-3)
    if fill_val is not None:
        batch_point_feats_tgt[invalid_mask] = fill_val
        assert torch.allclose(batch_point_feats_tgt[invalid_mask],
                              batch_point_feats[invalid_mask].cpu(), atol=1e-3, rtol=1e-3)


def test_gather_from():
    for num_feats in [(9,), tuple()]:
        for num_act_voxel in [65536, 32768]:
            for has_invalid_pts in [True, False]:
                for fill_val in [None, 6.9]:
                    print(num_feats, num_act_voxel, has_invalid_pts, fill_val)
                    _gather_from(num_max_voxel=65536, num_feats=num_feats,
                                 num_act_voxel=num_act_voxel, num_pts=200000,
                                 has_invalid_pts=has_invalid_pts, fill_val=fill_val)
