import torch
import numpy as np


class ScatterTo(torch.autograd.Function):
    """
  /**
   * IO Part:
   *    Input:
   *        0: batchPointFeats      [float]     [mMaxNumActIn, inChannels]
   *        1: scatterTo            [int32]     [mMaxNumActIn]
   *        2: scatterCount         [int32]     [mMaxNumActOut]
   *    Output:
   *        0: reducedFeats         [float]     [mMaxNumActOut, inChannels]
   * */
    """

    @staticmethod
    def forward(ctx,
                batch_point_feats: torch.Tensor,
                scatter_to: torch.Tensor,
                scatter_count: torch.Tensor,
                reduce_type: int) -> torch.Tensor:
        _, num_feats = batch_point_feats.shape
        num_coors = scatter_count.shape[0]
        reduces_feats = batch_point_feats.new_zeros((num_coors, num_feats))
        return reduces_feats

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 batch_point_feats: torch._C.Value,
                 scatter_to: torch._C.Value,
                 scatter_count: torch._C.Value,
                 reduce_type) -> torch._C.Value:
        data = np.array([reduce_type], dtype=np.int8).tobytes()
        return g.op('TRT_PluginV2',
                    batch_point_feats, scatter_to, scatter_count,
                    name_s=b'ScatterTo', data_s=data,
                    namespace_s=b'', version_s=b'1.0',
                    outputs=1)


scatter_to = ScatterTo.apply
