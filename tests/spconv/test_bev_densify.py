from symfun.ops.spconv import BEVDensify
from symfun.trt_utils import TRTPluginModule
import torch


def _densify(num_act_in, num_max_voxel, spatial_shape, feats_dtype):
    in_feats = torch.randn((num_max_voxel, 128), dtype=feats_dtype)
    in_spatial_shape = torch.empty(spatial_shape, dtype=torch.int32)
    db, _, dz, dy, dx = spatial_shape
    in_coors = torch.empty([0, 4], dtype=torch.int32)
    while in_coors.shape[0] < num_act_in:
        x = torch.randint(low=0, high=dx, size=(num_max_voxel, 1), dtype=torch.int32)
        y = torch.randint(low=0, high=dy, size=(num_max_voxel, 1), dtype=torch.int32)
        z = torch.randint(low=0, high=dz, size=(num_max_voxel, 1), dtype=torch.int32)
        b = torch.randint(low=0, high=db, size=(num_max_voxel, 1), dtype=torch.int32)
        _in_coors = torch.cat([b, z, y, x], dim=-1).unique(dim=0)
        in_coors = torch.cat([in_coors, _in_coors], dim=0).unique(dim=0)
    in_coors = in_coors[:num_act_in]
    num_act_in = torch.tensor([num_act_in], dtype=torch.int32)
    bev_shape = [db, 128, dz, dy, dx]
    gt_bev_feats = torch.zeros(bev_shape, dtype=feats_dtype)
    _b, _z, _y, _x = in_coors[:num_act_in].long().T
    gt_bev_feats[_b, :, _z, _y, _x] = in_feats[:num_act_in]
    gt_bev_feats = gt_bev_feats.reshape(db, 128 * dz, dy, dx)
    bev_feats = TRTPluginModule.forward(
        BEVDensify,
        input_tensors=(in_feats, in_coors, num_act_in, in_spatial_shape),
        configs=[128 * dz])
    assert torch.allclose(gt_bev_feats, bev_feats.cpu())


def test_densify_0():
    _densify(num_act_in=10000, num_max_voxel=10000, spatial_shape=(2, 0, 2, 128, 128), feats_dtype=torch.float32)


def test_densify_1():
    _densify(num_act_in=100, num_max_voxel=10000, spatial_shape=(2, 0, 2, 128, 128), feats_dtype=torch.float32)


def test_densify_2():
    _densify(num_act_in=0, num_max_voxel=10000, spatial_shape=(2, 0, 2, 128, 128), feats_dtype=torch.float32)


def test_densify_3():
    _densify(num_act_in=10000, num_max_voxel=10000, spatial_shape=(2, 0, 2, 128, 128), feats_dtype=torch.float16)


def test_densify_4():
    _densify(num_act_in=100, num_max_voxel=10000, spatial_shape=(2, 0, 2, 128, 128), feats_dtype=torch.float16)


def test_densify_5():
    _densify(num_act_in=0, num_max_voxel=10000, spatial_shape=(2, 0, 2, 128, 128), feats_dtype=torch.float16)
