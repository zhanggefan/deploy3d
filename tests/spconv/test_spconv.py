from deploy3d.symfun.ops.spconv import SPConvIdx3d, SPConvMM
from deploy3d.symfun.trt_utils import TRTPluginModule
import torch
from spconv.pytorch.ops import get_indice_pairs, indice_conv
from spconv.pytorch.core import ConvAlgo


def _make_unique_coors(num, shape):
    coors = torch.empty([0, 4], dtype=torch.int32)
    while coors.shape[0] < num:
        x = torch.randint(
            low=0, high=shape[-1], size=(num, 1), dtype=torch.int32)
        y = torch.randint(
            low=0, high=shape[-2], size=(num, 1), dtype=torch.int32)
        z = torch.randint(
            low=0, high=shape[-3], size=(num, 1), dtype=torch.int32)
        b = torch.randint(low=0, high=shape[0], size=(
            num, 1), dtype=torch.int32)
        _coors = torch.cat([b, z, y, x], dim=-1).unique(dim=0)
        coors = torch.cat([coors, _coors], dim=0).unique(dim=0)
    return coors[torch.randperm(len(coors))[:num]]


def _get_sequence(dst_coors, src_coors):
    dst_coors_idx = {tuple(c): idx for idx,
                     c in enumerate(dst_coors.cpu().tolist())}
    return [dst_coors_idx[tuple(c)] for c in src_coors.cpu().tolist()]


def test_spconv_index():
    in_spatial_shape = torch.empty([2, 0, 120, 90, 8], dtype=torch.int32)
    num_act_in = torch.tensor([20000], dtype=torch.int32)
    max_num_act_out = 50000
    in_coors = _make_unique_coors(num_act_in[0].item(), in_spatial_shape.shape)
    (gt_out_coors, gt_index, gt_index_buf_len) = get_indice_pairs(
        indices=in_coors.cuda(),
        batch_size=in_spatial_shape.shape[0],
        spatial_shape=in_spatial_shape.shape[2:],
        algo=ConvAlgo.Native,
        ksize=(3, 3, 3),
        stride=(2, 2, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
        out_padding=(0, 0, 0),
        subm=False,
        transpose=False)

    (index, index_buf_len, out_coors, num_act_out,
     out_spatial_shape) = TRTPluginModule.forward(
        SPConvIdx3d,
        input_tensors=(in_coors,
                       num_act_in,
                       in_spatial_shape),
        configs=((3, 3, 3),  # kernel_size
                 (2, 2, 1),  # stride
                 (1, 1, 1),  # padding
                 (1, 1, 1),  # dilation
                 (0, 0, 0),  # out_padding
                 max_num_act_out,  # max_num_act_out
                 False,  # subm
                 False))  # transpose

    seq = _get_sequence(out_coors[:num_act_out[0].item()], gt_out_coors)
    up_feats = torch.randn([max_num_act_out, 32], dtype=torch.float32)
    gt_up_feats = up_feats[seq]
    weights = torch.randn([3, 3, 3, 16, 32], dtype=torch.float32)
    bias = torch.randn([16], dtype=torch.float32)
    gt_out_feats = indice_conv(features=gt_up_feats.cuda(),
                               filters=weights.cuda(),
                               indice_pairs=gt_index.cuda(),
                               indice_pair_num=gt_index_buf_len.cuda(),
                               num_activate_out=num_act_in[0].item(),
                               inverse=True,
                               algo=ConvAlgo.Native)
    gt_out_feats += bias[None, :].cuda()

    out_feats = TRTPluginModule.forward(
        SPConvMM,
        input_tensors=(up_feats,
                       num_act_in,
                       num_act_out,
                       index,
                       index_buf_len),
        configs=((3, 3, 3),  # kernel_size
                 32,  # in_channels
                 16,  # out_channels
                 max_num_act_out,  # max_num_act_out
                 False,  # subm
                 True,  # inverse
                 weights.reshape(27, 16, 32).permute(
                     0, 2, 1).reshape(-1).numpy(),
                 bias.numpy()))  # bias
    out_feats = out_feats[:num_act_in[0].item()]
    assert torch.allclose(out_feats, gt_out_feats, atol=1e-1, rtol=1e-2)
