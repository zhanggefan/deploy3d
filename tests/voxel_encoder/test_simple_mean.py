from symfun.ops.voxel_encoder import SimpleMeanEncoder
from symfun.trt_utils import TRTPluginModule
import torch
import torch_scatter


def _simple_mean(max_num_act_out):
    x = torch.randn((200000, 1), dtype=torch.float32) * 50
    y = torch.randn((200000, 1), dtype=torch.float32) * 50
    z = torch.randn((200000, 1), dtype=torch.float32) * 2 + 1
    other = torch.randn((200000, 1), dtype=torch.float32)  # 2
    batch_point_feats = torch.cat([x, y, z, other], dim=-1)
    batch_indices = torch.randint(0, 2, (200000,), dtype=torch.int32)
    in_spatial_shape = torch.empty((2, 0, 41, 1536, 1536), dtype=torch.int32)
    voxel_config = torch.tensor([-76.8, -76.8, -2, 0.1, 0.1, 0.15])

    size = torch.tensor(in_spatial_shape.shape)[[4, 3, 2]]
    step, minimum = voxel_config[3:], voxel_config[:3]
    gt_coors_f = (batch_point_feats[:, :3] - minimum) / step

    mask = (gt_coors_f - gt_coors_f.round()).abs().amin(-1) > 1e-3
    batch_point_feats = batch_point_feats[mask]
    batch_indices = batch_indices[mask]
    gt_coors_f = gt_coors_f[mask]

    gt_coors = gt_coors_f.int()
    gt_coors_valid = gt_coors.lt(size.unsqueeze(0)).logical_and(gt_coors.ge(0)).all(dim=-1)
    gt_coors = torch.cat([batch_indices.unsqueeze(-1), gt_coors[:, [2, 1, 0]]], dim=-1)
    gt_coors[gt_coors_valid.logical_not()] = -1
    gt_coors, gt_scatter_to = gt_coors.unique(sorted=True, return_inverse=True, dim=0)
    if not gt_coors_valid.all():
        gt_coors = gt_coors[1:]
        gt_scatter_to -= 1
    gt_voxel_feats = torch_scatter.scatter_mean(batch_point_feats[gt_scatter_to.ge(0)],
                                                index=gt_scatter_to[gt_scatter_to.ge(0)], dim=0)

    (voxel_feats, out_coors, num_act_out) = TRTPluginModule.forward(
        SimpleMeanEncoder,
        input_tensors=(batch_point_feats, batch_indices,
                       voxel_config, in_spatial_shape,),
        configs=[max_num_act_out])

    # unique-ness of out_coors
    num_act_out = num_act_out.cpu()
    voxel_feats = voxel_feats.cpu()[:num_act_out]
    out_coors = out_coors.cpu()[:num_act_out]
    out_coors_tuple = [tuple(c) for c in out_coors.tolist()]
    assert len(out_coors_tuple) == len(set(out_coors_tuple)) == min(max_num_act_out, len(gt_coors))

    # get subset index
    gt_coors_tuple = [tuple(c) for c in gt_coors.tolist()]
    gt_coors_dict = {c: idx for idx, c in enumerate(gt_coors_tuple)}
    permute_index = torch.tensor([gt_coors_dict[c]
                                  for c in out_coors_tuple], dtype=torch.long)

    assert torch.allclose(voxel_feats.cpu(), gt_voxel_feats[permute_index], atol=1e-3, rtol=1e-3)
    assert (out_coors == gt_coors[permute_index]).all()


def test_simple_mean_0():
    _simple_mean(max_num_act_out=200000)


def test_simple_mean_1():
    _simple_mean(max_num_act_out=100)
