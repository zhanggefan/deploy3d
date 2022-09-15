from deploy3d.symfun.ops.point_voxel_mapping import ScatterTo
from deploy3d.symfun.trt_utils import TRTPluginModule
import torch
import torch_scatter


def _scatter_to(reduce_type, num_max_voxel, num_feats, num_act_voxel, num_pts, has_invalid_pts):
    batch_point_feats = torch.randn((num_pts, num_feats), dtype=torch.float32)
    scatter_to_base = torch.arange(-1 if has_invalid_pts else 0, num_act_voxel, dtype=torch.int32)
    assert num_pts - scatter_to_base.shape[0] >= 0
    scatter_to_extra = torch.randint(low=-1 if has_invalid_pts else 0, high=num_act_voxel,
                                     size=(num_pts - scatter_to_base.shape[0],), dtype=torch.int32)
    scatter_to = torch.cat([scatter_to_base, scatter_to_extra])
    scatter_to = scatter_to[torch.randperm(scatter_to.shape[0])]
    scatter_to_unique, _scatter_count = scatter_to.unique(sorted=True, return_counts=True)
    _scatter_count = _scatter_count[scatter_to_unique.ge(0)].int()
    scatter_to_unique = scatter_to_unique[scatter_to_unique.ge(0)]
    scatter_count = torch.empty([num_max_voxel], dtype=torch.int32)
    scatter_count[scatter_to_unique.long()] = _scatter_count

    reduced_feats = TRTPluginModule.forward(
        ScatterTo,
        input_tensors=(batch_point_feats,
                       scatter_to,
                       scatter_count),
        configs=(0 if reduce_type == 'max' else 1,)
    )
    if reduce_type == 'max':
        reduces_feats_tgt, _ = torch_scatter.scatter_max(src=batch_point_feats[scatter_to.ge(0)],
                                                         index=scatter_to[scatter_to.ge(0)].long(),
                                                         dim=0)
    else:
        reduces_feats_tgt = torch_scatter.scatter_mean(src=batch_point_feats[scatter_to.ge(0)],
                                                       index=scatter_to[scatter_to.ge(0)].long(),
                                                       dim=0)
    assert torch.allclose(reduces_feats_tgt, reduced_feats[:num_act_voxel].cpu(), atol=1e-3, rtol=1e-3)


def test_scatter_to_1():
    _scatter_to(
        reduce_type='max',
        num_max_voxel=65536,
        num_feats=9,
        num_act_voxel=65536, num_pts=200000, has_invalid_pts=True)


def test_scatter_to_2():
    _scatter_to(
        reduce_type='max',
        num_max_voxel=65536,
        num_feats=9,
        num_act_voxel=32768, num_pts=200000, has_invalid_pts=True)


def test_scatter_to_3():
    _scatter_to(
        reduce_type='max',
        num_max_voxel=65536,
        num_feats=9,
        num_act_voxel=65536, num_pts=200000, has_invalid_pts=False)


def test_scatter_to_4():
    _scatter_to(
        reduce_type='max',
        num_max_voxel=65536,
        num_feats=9,
        num_act_voxel=32768, num_pts=200000, has_invalid_pts=False)


def test_scatter_to_5():
    _scatter_to(
        reduce_type='mean',
        num_max_voxel=65536,
        num_feats=9,
        num_act_voxel=65536, num_pts=200000, has_invalid_pts=True)


def test_scatter_to_6():
    _scatter_to(
        reduce_type='mean',
        num_max_voxel=65536,
        num_feats=9,
        num_act_voxel=32768, num_pts=200000, has_invalid_pts=True)


def test_scatter_to_7():
    _scatter_to(
        reduce_type='mean',
        num_max_voxel=65536,
        num_feats=9,
        num_act_voxel=65536, num_pts=200000, has_invalid_pts=False)


def test_scatter_to_8():
    _scatter_to(
        reduce_type='mean',
        num_max_voxel=65536,
        num_feats=9,
        num_act_voxel=32768, num_pts=200000, has_invalid_pts=False)
