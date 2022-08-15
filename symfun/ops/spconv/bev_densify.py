import torch
import numpy as np


class BEVDensify(torch.autograd.Function):
    """
  /**
   * IO Part:
   *    Input:
   *        0: inFeats          [float/half]    [mMaxNumActIn, inChannels]
   *        1: inCoors          [int32]         [mMaxNumActIn, NDim + 1]
   *        2: numActIn         [int32]         [1]
   *        3: inSpatialShape   [void]          [B, 0, Z, Y, X]
   *    Output:
   *        0: outFeatMaps      [float/half]    [B, inChannels*Z, Y, X]
   * */
    """

    @staticmethod
    def forward(ctx,
                in_feats: torch.Tensor,
                in_coors: torch.Tensor,
                num_act_in: torch.Tensor,
                in_spatial_shape: torch.Tensor,
                out_channels) -> torch.Tensor:
        batch_size = in_spatial_shape.shape[0]
        ydim, xdim = in_spatial_shape.shape[-2:]
        bev = in_feats.new_zeros((batch_size, out_channels, ydim, xdim))
        return bev

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 in_feats: torch._C.Value,
                 in_coors: torch._C.Value,
                 num_act_in: torch._C.Value,
                 in_spatial_shape: torch._C.Value,
                 out_channels) -> torch._C.Value:
        data = np.array([out_channels], dtype=np.int32).tobytes()
        return g.op('TRT_PluginV2',
                    in_feats, in_coors, num_act_in, in_spatial_shape,
                    name_s=b"BEVDensify", data_s=data,
                    namespace_s=b'', version_s=b'1.0', outputs=1)


bev_densify = BEVDensify.apply
