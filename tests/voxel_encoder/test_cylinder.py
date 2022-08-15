from symfun.ops.voxel_encoder import CylinderEncoder
from symfun.trt_utils import TRTPluginModule
import torch
import numpy as np


def _cylinder(max_num_act_out):
    x = torch.randn((200000, 1), dtype=torch.float32) * 100 - 50
    y = torch.randn((200000, 1), dtype=torch.float32) * 100 - 50
    z = torch.randn((200000, 1), dtype=torch.float32) * 3 + 1
    other = torch.randn((200000, 1), dtype=torch.float32)  # 2
    batch_point_feats = torch.cat([x, y, z, other], dim=-1)
    batch_indices = torch.randint(0, 2, (200000,), dtype=torch.int32)
    in_spatial_shape = torch.empty((2, 0, 480, 360, 32), dtype=torch.int32)
    cylinder_config = torch.tensor([-2, -np.pi, 0, 4, np.pi, 50])

    gt_rho = (x * x + y * y).sqrt()
    gt_phi = torch.atan2(y, x)
    gt_polar = torch.cat([z, gt_phi, gt_rho], dim=-1)

    size = torch.tensor(in_spatial_shape.shape)[[4, 3, 2]]
    max, min = cylinder_config[3:], cylinder_config[:3]
    span = (max - min)
    step = span / size
    gt_coors = ((gt_polar - min) / step).int()
    gt_coor_z, gt_coor_phi, gt_coor_rho = gt_coors.split(1, -1)
    gt_coor_z.clamp_(0, size[0]-1)
    gt_coor_phi.clamp_(0, size[1]-1)
    gt_coor_rho.clamp_(0, size[2]-1)
    gt_coors = torch.cat([batch_indices.unsqueeze(-1),
                          gt_coor_rho, gt_coor_phi, gt_coor_z], dim=-1)
    gt_coors, gt_scatter_to, gt_counts = gt_coors.unique(sorted=True, return_inverse=True,
                                                         return_counts=True, dim=0)

    gt_pts_ctr = (torch.cat([gt_coor_rho, gt_coor_phi,
                  gt_coor_z],  dim=-1) + 0.5) * step + min

    gt_pts_diff = gt_polar - gt_pts_ctr
    pts_feats_gt = torch.cat([gt_polar, gt_pts_diff, x, y, other], dim=-1)

    (pts_feats, scatter_to, scatter_count, out_coors, num_act_out) = TRTPluginModule.forward(
        CylinderEncoder,
        input_tensors=(batch_point_feats, batch_indices,
                       cylinder_config, in_spatial_shape,),
        configs=[max_num_act_out])
    # unique-ness of out_coors
    num_act_out = num_act_out.cpu()
    out_coors = out_coors.cpu()[:num_act_out]
    scatter_count = scatter_count.cpu()[:num_act_out]
    scatter_to = scatter_to.cpu()
    out_coors_tuple = [tuple(c) for c in out_coors.tolist()]
    assert len(out_coors_tuple) == len(set(out_coors_tuple))

    # get subset index
    gt_coors_tuple = [tuple(c) for c in gt_coors.tolist()]
    gt_coors_dict = {c: idx for idx, c in enumerate(gt_coors_tuple)}
    permute_index = torch.tensor([gt_coors_dict[c]
                                 for c in out_coors_tuple], dtype=torch.long)
    permute_scatter_to = permute_index[scatter_to.long()]
    assert (permute_scatter_to[scatter_to.ge(0)] ==
            gt_scatter_to[scatter_to.ge(0)]).all()
    assert (scatter_count == gt_counts[permute_index]).all()


def test_cylinder_0():
    _cylinder(max_num_act_out=200000)


def test_cylinder_1():
    _cylinder(max_num_act_out=100)
