import torch
import numpy as np
from typing import Union


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
                scatter_to: torch.Tensor,
                fill_val: Union[float, None] = None) -> torch.Tensor:
        num_pts = scatter_to.shape[0]
        batch_point_feats = reduces_feats.new_zeros(
            [num_pts, *reduces_feats.shape[1:]])
        return batch_point_feats

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 reduces_feats: torch._C.Value,
                 scatter_to: torch._C.Value,
                 fill_val: Union[float, None] = None) -> torch._C.Value:
        data = np.array([fill_val is not None], dtype=np.bool).tobytes()
        data += np.array([fill_val], dtype=np.float32).tobytes()
        return g.op('TRT_PluginV2',
                    reduces_feats, scatter_to,
                    name_s=b'GatherBack', data_s=data,
                    namespace_s=b'', version_s=b'2.0',
                    outputs=1)


gather_back = GatherBack.apply
