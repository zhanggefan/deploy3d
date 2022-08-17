from typing import Union
import torch
import numpy as np


class SPConvMM(torch.autograd.Function):
    """
    /**
     * IO Part:
     *    Input:
     *        0: inFeats              [float/half]    [mMaxNumActIn, inChannels]
     *        1: numActIn             [int32]         [1]
     *        2: numActOut            [int32]         [1]
     *        3: index                [int32]         [3, kVol * mMaxNumActIn]
     *        4: (numBuf, bufSegLen)  [int32]         [1 + kVol]
     *    Output:
     *        0: outFeats             [float/half]    [mMaxNumActOut, outChannels]
     * */
    """

    @staticmethod
    def forward(ctx,
                in_feats: torch.Tensor,
                num_act_in: torch.Tensor,
                num_act_out: torch.Tensor,
                index: torch.Tensor,
                index_buf_len: torch.Tensor,
                kernel_size: Union[tuple, list],
                in_channels: int,
                out_channels: int,
                max_num_act_out: int,
                subm: bool,
                inverse: bool,
                weight: np.ndarray,
                bias: Union[None, np.ndarray] = None) -> torch.Tensor:
        out_feats = in_feats.new_zeros((max_num_act_out, out_channels))
        return out_feats

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 in_feats: torch._C.Value,
                 num_act_in: torch._C.Value,
                 num_act_out: torch._C.Value,
                 index: torch._C.Value,
                 index_buf_len: torch._C.Value,
                 kernel_size: Union[tuple, list],
                 in_channels: int,
                 out_channels: int,
                 max_num_act_out: int,
                 subm: bool,
                 inverse: bool,
                 weight: np.ndarray,
                 bias: Union[None, np.ndarray] = None) -> torch._C.Value:
        data = np.array([int(np.prod(kernel_size))], dtype=np.int32).tobytes()
        data += np.array([in_channels], dtype=np.int32).tobytes()
        data += np.array([out_channels], dtype=np.int32).tobytes()
        data += np.array([max_num_act_out], dtype=np.int32).tobytes()
        data += np.array([subm], dtype=np.bool).tobytes()
        data += np.array([inverse], dtype=np.bool).tobytes()
        data += np.array([bias is not None], dtype=np.bool).tobytes()
        data += weight.tobytes()
        if bias is not None:
            data += bias.tobytes()
        return g.op('TRT_PluginV2',
                    in_feats, num_act_in, num_act_out, index, index_buf_len,
                    name_s=b"SPConvMM", data_s=data, namespace_s=b'', version_s=b'2.0', outputs=1)


spconv_mm = SPConvMM.apply
