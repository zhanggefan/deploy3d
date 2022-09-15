import torch
import numpy as np
from deploy3d.symfun.ops.spconv import spconv_mm, spconv_index
from deploy3d.symfun.trt_utils import TRTPluginModule
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
        if self.conv.inverse:
            num_act_in, num_act_out = num_act_out, num_act_in
            out_coors, in_coors = in_coors, out_coors
            out_spatial_shape, in_spatial_shape = in_spatial_shape, out_spatial_shape

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
        return out_feats, out_coors, num_act_out, out_spatial_shape, index_dict


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
        
        # upA = self.up_subm(upA)
        return upA
        
        '''
        tmp = upA[0] + skip[0]
        upA = (tmp, skip[1], skip[2], skip[3], upA[4])

        upE = self.conv1(upA)

        upE = self.conv2(upE)

        upE = self.conv3(upE)

        return upE
        '''

class MiddleEncoder(torch.nn.Module):
    def __init__(self, middle_encoder, max_num_act_out, num_layers):
        super(MiddleEncoder, self).__init__()

        upBlocks = torch.nn.ModuleList()
        for i in range(num_layers):  # i = 0, 1, 2, 3, stride = 8, 4, 2, 1
            stride = pow(2, num_layers - i)
            indice_key = 'res_pool_' + str(i)
            upBlocks.append(
                UpBlock(middle_encoder.upBlocks[i], max_num_act_out, stride, indice_key))
        self.upBlocks = upBlocks

    def forward(self, x, skip):
        index_dict = dict()
        
        index = x.indices.new_zeros((3, 1000))
        index_buf_len = x.indices.new_zeros((1 + 9,))
        # index, index_buf_len, out_coors, num_act_out, out_spatial_shape, in_coors, num_act_in, in_spatial_shape
        
        
        num_act_in = torch.tensor([x.indices.shape[0]], dtype=torch.int32)
        in_spatial_shape = torch.empty((1, 0) + tuple(x.spatial_shape), dtype=torch.int32)
        
        num_act_out = torch.tensor([skip.indices.shape[0]], dtype=torch.int32)
        out_spatial_shape = torch.empty((1, 0) + tuple(skip.spatial_shape), dtype=torch.int32)
        
        index_dict['res_pool_1'] = (index, index_buf_len, x.indices, num_act_in, in_spatial_shape, 
                                    skip.indices, num_act_out, out_spatial_shape)
        

        upe = (x.features, x.indices, num_act_in, in_spatial_shape, index_dict)
        
        downb = (skip.features, skip.indices, num_act_out, out_spatial_shape, index_dict)
        
        upBlock = self.upBlocks[0]
        upe = upBlock(upe, downb)

        return upe[0], upe[1], upe[2], upe[3]
    


def get_permute_indexs(out_coors, indices, num_act_out):
    # assert out_coors.shape[0] == indices.shape[0]

    gt_coors_tuple = [tuple(c) for c in indices.tolist()]
    gt_coors_dict = {c: idx for idx, c in enumerate(gt_coors_tuple)}
    permute_index = torch.tensor([gt_coors_dict[tuple(c)]
                                  for c in out_coors.tolist()], dtype=torch.long)

    return permute_index


def test_middle_encoder():
    model_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/model_definition_1.pth'
    
    upBlock_1_up_1_path = '/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/upBlock_1_up_1.pth'
    upBlock_1_up_1 = torch.load(upBlock_1_up_1_path)

    x, skip = torch.load('/deepdata/cc_work_dirs/3d/cylinder3d_semantic_cowa_debug/0812_spconv2.x/upBlock_1_inputs.pth')
        
    model = torch.load(model_path)

    num_layers = 3
    num_act_in = 102124
    middle_encoder = MiddleEncoder(
        model.pts_middle_encoder, num_act_in * 1, num_layers).cuda()
    middle_encoder = revert_sync_batchnorm(middle_encoder)
    middle_encoder.eval()
    # middle_encoder.float()

    # in_feats, in_coors, num_act_in, in_spatial_shape
    logits = TRTPluginModule.forward(middle_encoder,
                                     input_tensors=(x,
                                                    skip,))
    
    out_coors = logits[1]
    num_act_out = logits[2]
    indices = upBlock_1_up_1.indices
    out_coors = out_coors[:num_act_out]
    permute_indexs = get_permute_indexs(out_coors, indices, num_act_out)
    logits = logits[0][:num_act_out]

    # (logits - upBlock_1_up_1.features[permute_indexs]).max()
    assert torch.allclose(logits.float().cpu(), upBlock_1_up_1.features[permute_indexs].float().cpu(), atol=1e-1, rtol=1e-1)