from typing import Tuple
import torch
import numpy as np


class SimpleMeanEncoder(torch.autograd.Function):
    """
  /**
   * IO Part:
   *    Input:
   *        0: batchPointFeats      [float]     [mMaxNumActIn, inChannels]
   *        1: batchIndices         [int32]     [mMaxNumActIn]
   *        2. voxelConfig          [float]     [6]
   *        3: inSpatialShape       [void]      [B, 0, Z, Y, X]
   *    Output:
   *        0: outFeats             [float]     [mMaxNumActOut, inChannels]
   *        1: outCoors             [int32]     [mMaxNumActOut, 4]
   *        2: numActOut            [int32]     [1]
   * */
    """

    @staticmethod
    def forward(ctx,
                batch_point_feats: torch.Tensor,
                batch_indices: torch.Tensor,
                voxel_config: torch.Tensor,
                in_spatial_shape: torch.Tensor,
                max_num_act_out: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        out_feats = batch_point_feats.new_zeros(
            (max_num_act_out, batch_point_feats.shape[-1]))
        out_coors = batch_indices.new_zeros((max_num_act_out, 4))
        num_act_out = batch_indices.new_zeros((1,))
        return out_feats, out_coors, num_act_out

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 batch_point_feats: torch._C.Value,
                 batch_indices: torch._C.Value,
                 voxel_config: torch._C.Value,
                 in_spatial_shape: torch._C.Value,
                 max_num_act_out: int) -> Tuple[
        torch._C.Value, torch._C.Value, torch._C.Value]:
        data = np.array([max_num_act_out], dtype=np.int32).tobytes()
        return g.op('TRT_PluginV2',
                    batch_point_feats, batch_indices, voxel_config,
                    in_spatial_shape,
                    name_s=b'SimpleMeanEncoder', data_s=data, namespace_s=b'',
                    version_s=b'1.0', outputs=3)


simple_mean_encoder = SimpleMeanEncoder.apply
