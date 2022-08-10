import torch


class GatherBack(torch.autograd.Function):
    """
  /**
   * IO Part:
   *    Input:
   *        0: reducedFeats         [float]     [mMaxNumActOut, inChannels]
   *        1: scatterTo            [int32]     [mMaxNumActIn]
   *    Output:
   *        0: batchPointFeats      [float]     [mMaxNumActIn, inChannels]
   * */
    """

    @staticmethod
    def forward(ctx,
                reduces_feats: torch.Tensor,
                scatter_to: torch.Tensor) -> torch.Tensor:
        num_pts = scatter_to.shape
        _, num_feats = reduces_feats.shape
        batch_point_feats = reduces_feats.new_zeros((num_pts, num_feats))
        return batch_point_feats

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 reduces_feats: torch._C.Value,
                 scatter_to: torch._C.Value) -> torch._C.Value:
        return g.op('TRT_PluginV2',
                    reduces_feats, scatter_to,
                    name_s=b'GatherBack', data_s=b'',
                    namespace_s=b'', version_s=b'1.0',
                    outputs=1)


gather_back = GatherBack.apply
