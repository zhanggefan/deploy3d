from symfun.ops.voxel_encoder import CylinderEncoder
from symfun.trt_utils import TRTPluginModule
import torch
import numpy as np


def test_cylinder():
    x = torch.randn((200000, 1), dtype=torch.float32) * 100 - 50
    y = torch.randn((200000, 1), dtype=torch.float32) * 100 - 50
    z = torch.randn((200000, 1), dtype=torch.float32) * 3 + 1
    other = torch.randn((200000, 2), dtype=torch.float32)
    batch_point_feats = torch.cat([x, y, z, other], dim=-1)
    batch_indices = torch.randint(0, 2, (200000,), dtype=torch.int32)
    in_spatial_shape = torch.empty((2, 0, 480, 360, 32), dtype=torch.int32)
    cylinder_config = torch.tensor([-2, -np.pi, 0, 4, np.pi, 50])

    rho = (x * x + y * y).sqrt()
    phi = torch.atan2(y, x)
    polar = torch.cat([z, phi, rho], dim=-1)

    size = torch.tensor(in_spatial_shape.shape)[[4, 3, 2]]
    max, min = cylinder_config[3:], cylinder_config[:3]
    span = (max - min)
    step = span / size
    coor = ((polar - min) / step).int()
    coor_z, coor_phi, coor_rho = coor.split(1, -1)
    coor_z.clamp_(0, size[0]-1)
    coor_phi.clamp_(0, size[1]-1)
    coor_rho.clamp_(0, size[2]-1)
    coor = torch.cat([batch_indices.unsqueeze(-1),
                     coor_rho, coor_phi, coor_z], dim=-1)
    coor, inverse, counts = coor.unique(sorted=True, return_inverse=True,
                                        return_counts=True, dim=0)

    (pts_feats, scatter_to, scatter_count, out_coors, num_act_out) = TRTPluginModule.forward(
        CylinderEncoder,
        input_tensors=(batch_point_feats, batch_indices,
                       cylinder_config, in_spatial_shape,),
        configs=[1000])

    if len(out_coors) < len(coor):
        return

    out_coors = out_coors[:num_act_out].cpu()
    scatter_count = scatter_count[:num_act_out].cpu()

    out_unique, out_inverse, out_counts = out_coors.unique(
        sorted=True, return_inverse=True, return_counts=True, dim=0)
    assert (len(out_unique) == len(out_coors))
    assert (coor == out_unique).all()
    scatter_to_reorder = out_inverse[scatter_to.long()]
    scatter_count_reorder = scatter_count.clone()
    scatter_count_reorder[out_inverse] = scatter_count
    assert (scatter_to_reorder == inverse).all()
    assert (scatter_count_reorder == counts).all()


test_cylinder()
