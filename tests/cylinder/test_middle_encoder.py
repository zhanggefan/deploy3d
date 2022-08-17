import torch
import numpy as np
from symfun.ops.spconv import spconv_mm, spconv_index
from symfun.trt_utils import TRTPluginModule
from mmcv.cnn.utils import revert_sync_batchnorm


class InferSpConvModule(torch.nn.Module):
    def __init__(self, conv, bn, act, index_name, max_num_act_out):
        super(InferSpConvModule, self).__init__()
        if bn:
            self.weight, self.bias = self.fused_weight_bias(conv, bn)
        else:
            self.weight, self.bias = self.only_weight(conv)
        self.conv = conv
        self.bn = bn
        self.act = act
        self.index_name = index_name
        self.max_num_act_out = max_num_act_out

    @torch.no_grad()
    def fused_weight_bias(self, conv, bn):
        w = conv.weight
        std = bn.running_var.sqrt()
        m = bn.running_mean
        nw = bn.weight
        w = w / std[:, None, None, None, None] * nw[:, None, None, None, None]
        b = bn.bias - m / std * nw
        o, i = w.shape[0], w.shape[4]
        kvol = w.shape[1] * w.shape[2] * w.shape[3]
        w = w.reshape(o, kvol, i)
        w = w.permute(1, 2, 0)
        w = w.reshape(-1).contiguous()
        return w.float().cpu().numpy(), b.float().cpu().numpy()

    @torch.no_grad()
    def only_weight(self, conv):
        b = torch.zeros((conv.weight.shape[0]))

        w = conv.weight
        o, i = w.shape[0], w.shape[4]
        kvol = w.shape[1] * w.shape[2] * w.shape[3]
        w = w.reshape(o, kvol, i)
        w = w.permute(1, 2, 0)
        w = w.reshape(-1).contiguous()

        return w.float().cpu().numpy(), b.float().cpu().numpy()

    def forward(self, inp):
        in_feats, in_coors, num_act_in, in_spatial_shape, index_dict = inp
        if self.index_name in index_dict:
            indices = index_dict[self.index_name]
        else:
            assert not self.conv.inverse
            indices = spconv_index(
                in_coors,
                num_act_in,
                in_spatial_shape,
                self.conv.kernel_size,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.output_padding,
                self.max_num_act_out,
                self.conv.subm,
                self.conv.transposed)
            if self.conv.subm:
                indices = indices + (in_coors, num_act_in, in_spatial_shape)
            indices = indices + (in_coors, num_act_in, in_spatial_shape)
            index_dict[self.index_name] = indices
        
        index, index_buf_len, out_coors, num_act_out, out_spatial_shape, in_coors, num_act_in, in_spatial_shape = indices
        
        out_feats = spconv_mm(
            in_feats,
            num_act_in,
            num_act_out,
            index,
            index_buf_len,
            self.conv.kernel_size,
            self.conv.in_channels,
            self.conv.out_channels,
            self.max_num_act_out,
            self.conv.subm,
            self.conv.inverse,
            self.weight,
            self.bias)
        if self.act:
            out_feats = self.act(out_feats)
        
        if self.conv.inverse:
            num_act_in, num_act_out = num_act_out, num_act_in
            out_coors, in_coors = in_coors, out_coors
            out_spatial_shape, in_spatial_shape = in_spatial_shape, out_spatial_shape
        
        return out_feats, out_coors, num_act_out, out_spatial_shape, index_dict


class ResContextBlock(torch.nn.Module):
    def __init__(self, model, max_num_act_out):
        super(ResContextBlock, self).__init__()
        self.conv1 = InferSpConvModule(*model.conv1, 'pre_1', max_num_act_out)
        self.conv1_2 = InferSpConvModule(
            *model.conv1_2, 'pre_1_2', max_num_act_out)
        self.conv2 = InferSpConvModule(*model.conv2, 'pre_2', max_num_act_out)
        self.conv3 = InferSpConvModule(*model.conv3, 'pre_3', max_num_act_out)

    def forward(self, x):
        shortcut = self.conv1(x)
        
        shortcut = self.conv1_2(shortcut)

        resA = self.conv2(x)

        resA = self.conv3(resA)

        tmp = resA[0] + shortcut[0]

        return (tmp, resA[1], resA[2], resA[3], resA[4])
        

class ResBlock(torch.nn.Module):
    def __init__(self, model, max_num_act_out, stride, indice_key):
        super(ResBlock, self).__init__()

        self.conv1 = InferSpConvModule(
            *model.conv1, 'res1_' + str(stride), max_num_act_out)  # // stride)
        self.conv1_2 = InferSpConvModule(
            *model.conv1_2, 'res1_2_' + str(stride), max_num_act_out)  # // stride)
        self.conv2 = InferSpConvModule(
            *model.conv2, 'res2_' + str(stride), max_num_act_out)  # // stride)
        self.conv3 = InferSpConvModule(
            *model.conv3, 'res3_' + str(stride), max_num_act_out)  # // stride)
        self.pool = InferSpConvModule(
            model.pool, None, None, indice_key, max_num_act_out)  # // (2 * stride))

    def forward(self, x):
        shortcut = self.conv1(x)

        shortcut = self.conv1_2(shortcut)

        resA = self.conv2(x)

        resA = self.conv3(resA)

        tmp = resA[0] + shortcut[0]  # resA[0] = resA[0] + shortcut[0]
        resA = (tmp, resA[1], resA[2], resA[3], resA[4])

        resB = self.pool(resA)

        return resB, resA


class UpBlock(torch.nn.Module):
    def __init__(self, model, max_num_act_out, stride, indice_key):
        super(UpBlock, self).__init__()

        self.trans_dilao = InferSpConvModule(
            *model.trans_dilao, 'up_td_' + str(stride), max_num_act_out)  # // stride)
        self.conv1 = InferSpConvModule(
            *model.conv1, 'up_1_' + str(stride), max_num_act_out)  # // stride)
        self.conv2 = InferSpConvModule(
            *model.conv2, 'up_2_' + str(stride), max_num_act_out)  # // stride)
        self.conv3 = InferSpConvModule(
            *model.conv3, 'up_3_' + str(stride), max_num_act_out)  # // stride)
        self.up_subm = InferSpConvModule(
            model.up_subm, None, None, indice_key, max_num_act_out)  # // (stride / 2))

    def forward(self, x, skip):
        upA = self.trans_dilao(x)

        upA = self.up_subm(upA)
        
        tmp = upA[0] + skip[0]
        upA = (tmp, skip[1], skip[2], skip[3], upA[4])

        upE = self.conv1(upA)

        upE = self.conv2(upE)

        upE = self.conv3(upE)

        return upE        


class ReconBlock(torch.nn.Module):
    def __init__(self, model, max_num_act_out):
        super(ReconBlock, self).__init__()

        self.conv1 = InferSpConvModule(
            model.conv1[0][0], model.conv1[0][1], model.conv1[1], 'rec_1', max_num_act_out)
        self.conv1_2 = InferSpConvModule(
            model.conv1_2[0][0], model.conv1_2[0][1], model.conv1_2[1], 'rec_1_2', max_num_act_out)
        self.conv1_3 = InferSpConvModule(
            model.conv1_3[0][0], model.conv1_3[0][1], model.conv1_3[1], 'rec_1_3', max_num_act_out)

    def forward(self, x):
        shortcut = self.conv1(x)

        shortcut2 = self.conv1_2(x)

        shortcut3 = self.conv1_3(x)

        tmp = shortcut[0] + shortcut2[0] + shortcut3[0]
        shortcut = (tmp, shortcut[1], shortcut[2], shortcut[3], shortcut3[4])
        
        tmp = shortcut[0] * x[0]

        return (tmp, shortcut[1], shortcut[2], shortcut[3], shortcut[4])


class MiddleEncoder(torch.nn.Module):
    def __init__(self, middle_encoder, max_num_act_out, num_layers):
        super(MiddleEncoder, self).__init__()

        self.downCntx = ResContextBlock(
            middle_encoder.downCntx, max_num_act_out)

        indice_keys = []
        resBlocks = torch.nn.ModuleList()
        for i in range(num_layers):  # i = 0, 1, 2, 3, stride = 1, 2, 4, 8
            stride = pow(2, i)
            indice_key = 'res_pool_' + str(stride)
            indice_keys.append(indice_key)
            resBlocks.append(
                ResBlock(middle_encoder.resBlocks[i], max_num_act_out, stride, indice_key))
        self.resBlocks = resBlocks

        upBlocks = torch.nn.ModuleList()
        for i in range(num_layers):  # i = 0, 1, 2, 3, stride = 8, 4, 2, 1
            indice_key = indice_keys.pop()
            stride = pow(2, num_layers - i)
            upBlocks.append(
                UpBlock(middle_encoder.upBlocks[i], max_num_act_out, stride, indice_key))
        self.upBlocks = upBlocks

        self.ReconNet = ReconBlock(middle_encoder.ReconNet, max_num_act_out)

        self.logits = InferSpConvModule(
            middle_encoder.logits, None, None, 'logits', max_num_act_out)

    def forward(self, in_feats, in_coors, num_act_in, in_spatial_shape):
        index_dict = dict()

        # downcntx
        downc = self.downCntx(
            (in_feats, in_coors, num_act_in, in_spatial_shape, index_dict))
        
        # resblocks
        res = []
        for resBlock in self.resBlocks:
            downc, downb = resBlock(downc)
            res.append((downc, downb))
        
        # upblocks
        upe = res[-1][0]
        for upBlock in self.upBlocks:
            _, downb = res.pop()
            upe = upBlock(upe, downb)
        
        # reconnet
        up0e = self.ReconNet(upe)
        
        tmp = torch.cat((up0e[0], upe[0]), 1)
        up0e = (tmp, up0e[1], up0e[2], up0e[3], up0e[4])
        
        # logits
        logits = self.logits(up0e)
        return logits[0]


def get_permute_indexs(out_coors, indices):

    gt_coors_tuple = [tuple(c) for c in indices.tolist()]
    gt_coors_dict = {c: idx for idx, c in enumerate(gt_coors_tuple)}
    permute_index = torch.tensor([gt_coors_dict[tuple(c)]
                                  for c in out_coors.tolist()], dtype=torch.long)

    return permute_index


def test_middle_encoder():
    model_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/model_definition_1.pth'
    voxel_feats_out_coors_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/pfn_outs_1.pth'

    #########################################
    # downcntx_shortcut_1_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/downcntx_shortcut_1.pth'
    # downcntx_shortcut_1 = torch.load(downcntx_shortcut_1_path)

    # downcntx_resA_3_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/resA_3.pth'
    # downcntx_resA_3 = torch.load(downcntx_resA_3_path)

    # downcntx_out_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/downcntx_out.pth'
    # downcntx_out = torch.load(downcntx_out_path)

    # resBlock1_shortcut_1_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/resBlock1_shortcut_1.pth'
    # resBlock1_shortcut_1 = torch.load(resBlock1_shortcut_1_path)

    # resBlock3_pool_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/resBlock3_pool.pth'
    # resBlock3_pool = torch.load(resBlock3_pool_path)

    # upBlock_1_upA_1_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/upBlock_1_upA_1.pth'
    # upBlock_1_upA_1 = torch.load(upBlock_1_upA_1_path)

    # upBlock_1_up_1_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/upBlock_1_up_1.pth'
    # upBlock_1_up_1 = torch.load(upBlock_1_up_1_path)

    # upBlock_1_outputs_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/upBlock_1_outputs.pth'
    # upBlock_1_outputs = torch.load(upBlock_1_outputs_path)
    
    # upBlock_3_outputs_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/upBlock_3_outputs.pth'
    # upBlock_3_outputs = torch.load(upBlock_3_outputs_path)
    
    # reconnet_sum_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/reconnet_sum.pth'
    # reconnet_sum = torch.load(reconnet_sum_path)
    
    # reconnet_dot_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/reconnet_dot.pth'
    # reconnet_dot = torch.load(reconnet_dot_path)
    
    # cat_before_logits_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/cat_before_logits.pth'
    # cat_before_logits = torch.load(cat_before_logits_path) 
    
    logits_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/logits.pth'
    logits_target = torch.load(logits_path)
    
    # logits_dense_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/logits_dense.pth'
    # logits_dense = torch.load(logits_dense_path)
    #########################################

    model = torch.load(model_path)
    voxel_feats_gt, out_coors_gt = torch.load(voxel_feats_out_coors_path)
    in_spatial_shape = torch.empty((1, 0, 480, 360, 32), dtype=torch.int32)

    num_layers = 3
    num_act_in = 102124
    middle_encoder = MiddleEncoder(
        model.pts_middle_encoder, num_act_in * 1, num_layers).cuda()
    middle_encoder = revert_sync_batchnorm(middle_encoder)
    middle_encoder.eval()
    # middle_encoder.float()

    # in_feats, in_coors, num_act_in, in_spatial_shape
    logits = TRTPluginModule.forward(middle_encoder,
                                     input_tensors=(voxel_feats_gt,
                                                    out_coors_gt,
                                                    torch.tensor(
                                                        [num_act_in], dtype=torch.int32),
                                                    in_spatial_shape))

    assert torch.allclose(logits.float().cpu(), logits_target.features.float().cpu(), atol=1e-1, rtol=1e-1)
    
    '''
    out_coors = logits[1]
    num_act_out = logits[2]
    indices = upBlock_3_outputs.indices
    out_coors = out_coors[:num_act_out]
    permute_indexs = get_permute_indexs(out_coors, indices, num_act_out)
    logits = logits[0][:num_act_out]

    # (logits - upBlock_1_up_1.features[permute_indexs]).max()
    assert torch.allclose(logits.float().cpu(
    ), upBlock_3_outputs.features[permute_indexs].float().cpu(), atol=1e-1, rtol=1e-1)

    # logits_dense = logits_dense.squeeze().permute(1, 2, 3, 0)
    # assert torch.allclose(logits.float().cpu(), downcntx_shortcut_1.features().float().cpu(), atol=1e-1, rtol=1e-1)
    '''

# test_middle_encoder()
