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


def _test_spconv_index(ksize=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1), dilation=(1, 1, 1), out_padding=(0, 0, 0),
                       num_act_in=20000, in_spatial_shape=[2, 0, 120, 90, 8], max_num_act_in=40000,
                       max_num_act_out=50000, subm=False, transpose=False):
    in_spatial_shape = torch.empty(in_spatial_shape, dtype=torch.int32)
    num_act_in = torch.tensor([num_act_in], dtype=torch.int32)
    in_coors = torch.empty([max_num_act_in, 4], dtype=torch.int32, device='cuda')
    in_coors_ = _make_unique_coors(num_act_in[0].item(), in_spatial_shape.shape).cuda()
    in_coors[:num_act_in[0].item()] = in_coors_

    (gt_out_coors, gt_index, gt_index_buf_len) = get_indice_pairs(
        indices=in_coors_,
        batch_size=in_spatial_shape.shape[0],
        spatial_shape=in_spatial_shape.shape[2:],
        algo=ConvAlgo.Native,
        ksize=ksize,
        stride=stride,
        padding=padding,
        dilation=dilation,
        out_padding=out_padding,
        subm=subm,
        transpose=transpose)

    _index = TRTPluginModule.forward(
        SPConvIdx3d,
        input_tensors=(in_coors,
                       num_act_in,
                       in_spatial_shape),
        configs=(ksize,  # kernel_size
                 stride,  # stride
                 padding,  # padding
                 dilation,  # dilation
                 out_padding,  # out_padding
                 max_num_act_out,  # max_num_act_out
                 subm,  # subm
                 transpose))  # transpose

    if subm:
        (index, num_index) = _index
    else:
        (index, num_index, out_coors, num_act_out, out_spatial_shape) = _index

        out_coors = out_coors[:num_act_out.long()]

        gt_coors_idx = {tuple(c): idx for idx, c in enumerate(gt_out_coors.cpu().tolist())}
        for c in out_coors.cpu().tolist():
            assert tuple(c) in gt_coors_idx

        idx_in_gt = torch.tensor([gt_coors_idx[tuple(c)] for c in out_coors.cpu().tolist()]).cuda()
        assert len(idx_in_gt.unique()) == len(idx_in_gt)
        assert len(idx_in_gt) == min(len(gt_out_coors), max_num_act_out)
        gt_present = torch.zeros([gt_out_coors.shape[0]], dtype=torch.bool).cuda()
        gt_present[idx_in_gt] = True

    assert num_index % 128 == 0
    gather_in, scatter_out, kernel_offset = index
    kernel_segments = torch.cat([torch.nonzero(kernel_offset == _)[[0, -1]] for _ in range(27)], 1).T
    kernel_segments[:, 1] += 1

    assert ((kernel_segments[:, 0] % 128) == 0).all()

    is_padding = torch.logical_or(kernel_offset.ge(27), kernel_offset.lt(0))
    assert (gather_in[is_padding] == -1).all()
    assert (scatter_out[is_padding] == -1).all()
    for i, (st, ed) in enumerate(kernel_segments):
        gather_in_kernel_i = gather_in[st:ed]
        scatter_out_kernel_i = scatter_out[st:ed]
        sort_arg = gather_in_kernel_i.argsort()
        gather_in_kernel_i = gather_in_kernel_i[sort_arg]
        scatter_out_kernel_i = scatter_out_kernel_i[sort_arg]
        if not subm:
            scatter_out_kernel_i = idx_in_gt[scatter_out_kernel_i.long()]

        if subm:
            _ksize = torch.prod(torch.tensor(ksize))
            if i < _ksize // 2:
                gt_gather_scatter_kernel_i = gt_index[:, i][..., :gt_index_buf_len[i]]
            elif i > _ksize // 2:
                gt_gather_scatter_kernel_i = gt_index[:, _ksize - 1 - i][..., :gt_index_buf_len[_ksize - 1 - i]][[1, 0]]
            else:
                gt_gather_scatter_kernel_i = torch.arange(num_act_in[0].item()).cuda().int().view(1, -1).repeat(2, 1)

        else:
            gt_gather_scatter_kernel_i = gt_index[:, i][..., :gt_index_buf_len[i]]
        gt_gather_in_kernel_i = gt_gather_scatter_kernel_i[0]
        gt_scatter_out_kernel_i = gt_gather_scatter_kernel_i[1]
        if not subm:
            gt_scatter_out_kernel_i_present = gt_present[gt_scatter_out_kernel_i.long()]
            gt_gather_in_kernel_i = gt_gather_in_kernel_i[gt_scatter_out_kernel_i_present]
            gt_scatter_out_kernel_i = gt_scatter_out_kernel_i[gt_scatter_out_kernel_i_present]

        sort_arg = gt_gather_in_kernel_i.argsort()
        gt_gather_in_kernel_i = gt_gather_in_kernel_i[sort_arg]
        gt_scatter_out_kernel_i = gt_scatter_out_kernel_i[sort_arg]

        assert (gather_in_kernel_i == gt_gather_in_kernel_i).all()
        assert (scatter_out_kernel_i == gt_scatter_out_kernel_i).all()


def test_spconv_index():
    _test_spconv_index(ksize=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1), num_act_in=20000, max_num_act_in=20000,
                       max_num_act_out=50000, subm=False)
    _test_spconv_index(ksize=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1), num_act_in=20000, max_num_act_in=40000,
                       max_num_act_out=50000, subm=False)
    _test_spconv_index(ksize=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1), num_act_in=20000, max_num_act_in=20000,
                       max_num_act_out=5000, subm=False)
    _test_spconv_index(ksize=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1), num_act_in=20000, max_num_act_in=40000,
                       max_num_act_out=5000, subm=False)
    _test_spconv_index(ksize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), num_act_in=20000, max_num_act_in=20000,
                       max_num_act_out=50000, subm=True)
    _test_spconv_index(ksize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), num_act_in=20000, max_num_act_in=40000,
                       max_num_act_out=50000, subm=True)
    _test_spconv_index(ksize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), num_act_in=20000, max_num_act_in=20000,
                       max_num_act_out=5000, subm=True)
    _test_spconv_index(ksize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), num_act_in=20000, max_num_act_in=40000,
                       max_num_act_out=5000, subm=True)


if __name__ == '__main__':
    test_spconv_index()
