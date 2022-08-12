from symfun.ops.spconv import SpConvIdx3d
from symfun.trt_utils import TRTPluginModule
import torch
from spconv.pytorch.ops import get_indice_pairs
from spconv.pytorch.core import ConvAlgo


def test_spconv_index():
    in_coors = torch.empty((1000, 4), dtype=torch.int32)
    in_coors[:4] = torch.tensor([[0, 3, 5, 5], [0, 3, 6, 5], [0, 2, 8, 4], [
        1, 0, 8, 4]], dtype=torch.int32)
    num_act_in = torch.full((1,), 4, dtype=torch.int32)
    in_spatial_shape = torch.empty((2, 0, 4, 10, 10), dtype=torch.int32)

    (out_inds, pair, indice_num_per_loc) = get_indice_pairs(
        in_coors[:4],
        batch_size=2,
        spatial_shape=(4, 10, 10),
        algo=ConvAlgo.Native,
        ksize=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
        out_padding=(0, 0, 0),
        subm=False,
        transpose=False)

    (index, index_buf_len, out_coors, num_act_out,
     out_spatial_shape) = TRTPluginModule.forward(
        SpConvIdx3d,
        input_tensors=(in_coors,
                       num_act_in,
                       in_spatial_shape),
        configs=((3, 3, 3),  # kernel_size
                 (1, 1, 1),  # stride
                 (1, 1, 1),  # padding
                 (1, 1, 1),  # dilation
                 (0, 0, 0),  # out_padding
                 100,  # max_num_act_out
                 False,  # subm
                 False))  # transpose
    return
