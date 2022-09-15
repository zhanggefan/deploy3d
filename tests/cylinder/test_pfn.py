import torch
import numpy as np
from deploy3d.symfun.ops.voxel_encoder import cylinder_encoder
from deploy3d.symfun.ops.point_voxel_mapping import scatter_to, gather_back
from deploy3d.symfun.trt_utils import TRTPluginModule
from mmcv.cnn.utils import revert_sync_batchnorm


class PointFeatureNet(torch.nn.Module):
    def __init__(self, model, max_num_act_out=102124, reduce_type=0):
        super(PointFeatureNet, self).__init__()

        self.reduce_type = reduce_type
        self.max_num_act_out = max_num_act_out
        self.pfn_layers = model.pfn_layers
        self.post_reduce_layers = model.post_reduce_layers

    def forward(self, batch_point_feats, batch_indices, cylinder_config, in_spatial_shape):
        pts_feats, scatter_index, scatter_count, out_coors, num_act_out = cylinder_encoder(batch_point_feats,
                                                                                           batch_indices,
                                                                                           cylinder_config,
                                                                                           in_spatial_shape,
                                                                                           self.max_num_act_out)
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(pts_feats)
            voxel_feats = scatter_to(
                point_feats, scatter_index, scatter_count, self.reduce_type)
            if i != len(self.pfn_layers) - 1:
                feat_per_point = gather_back(voxel_feats, scatter_index)
                pts_feats = torch.cat([point_feats, feat_per_point], dim=1)

        voxel_feats = self.post_reduce_layers(voxel_feats)
        return voxel_feats, out_coors, num_act_out


def load_chencpoint(model, state_dict):
    new_state_dict = {}
    for key in model.state_dict().keys():
        old_key = 'pts_voxel_encoder.' + key
        new_state_dict[key] = state_dict[old_key]

    model.load_state_dict(new_state_dict)
    return model


def test_pfn():
    batch_point_feats = torch.load(
        '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/points.pth')[0]
    batch_indices = torch.zeros(
        [batch_point_feats.shape[0]], dtype=torch.int32)
    in_spatial_shape = torch.empty((1, 0, 480, 360, 32), dtype=torch.int32)
    cylinder_config = torch.tensor([-2, -np.pi, 0, 4, np.pi, 50])

    model = torch.load(
        '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/model_definition.pth')
    pfn_layer = PointFeatureNet(model.pts_voxel_encoder).cuda()
    # pfn_layer = load_chencpoint(pfn_layer, model.state_dict())
    pfn_layer = revert_sync_batchnorm(pfn_layer)
    pfn_layer.eval()
    pfn_layer.float()

    voxel_feats, out_coors, num_act_out = TRTPluginModule.forward(pfn_layer,
                                                                  input_tensors=(batch_point_feats.float(),
                                                                                 batch_indices,
                                                                                 cylinder_config,
                                                                                 in_spatial_shape))

    voxel_feats_gt, out_coors_gt = torch.load(
        '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/pfn_outs.pth')

    
    
    out_coors_tuple = [tuple(c) for c in out_coors.tolist()]
    gt_coors_tuple = [tuple(c) for c in out_coors_gt.tolist()]
    gt_coors_dict = {c: idx for idx, c in enumerate(gt_coors_tuple)}
    permute_index = torch.tensor([gt_coors_dict[c]
                                 for c in out_coors_tuple], dtype=torch.long)
    
    
    
    assert torch.allclose(voxel_feats_gt[permute_index].cpu(), voxel_feats.cpu(), atol=1e-2, rtol=1e-2)