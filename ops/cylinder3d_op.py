import torch
import numpy as np
import torch.nn.functional as F

class Inspect(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                tensor,
                name: str):
        return tensor

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 tensor,
                 name: str):
        data = name.encode()
        return g.op('TRT_PluginV2', tensor, name_s=b'Inspect', data_s=data, namespace_s=b'', version_s=b'1.0',
                    outputs=1)

# inspect = Inspect.apply
inspect = lambda tensor, name: tensor


class DynamicCylinder3dVoxelize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                points,
                cylinder_partition,
                cylinder_range):
        num_points = points.shape[0] * points.shape[1]
        res_points = points.new_zeros((num_points, 6), dtype = torch.float32)
        res_coors = points.new_zeros((num_points, 4), dtype = torch.int32)
        return res_points, res_coors

    @staticmethod
    def symbolic(g,
                 points,
                 cylinder_partition,
                 cylinder_range):
        data = np.array(cylinder_partition, dtype=np.int32).tobytes()
        data += np.array(cylinder_range, dtype=np.float32).tobytes()
        return g.op('TRT_PluginV2', points, name_s=b'DynamicCylinder3dVoxelize', data_s=data, namespace_s=b'', version_s=b'1.0',
                    outputs=2)

CylinderVoxelization = DynamicCylinder3dVoxelize.apply


class Voxelization(torch.nn.Module):
    def __init__(self, cylinder_partition, cylinder_range):
        super().__init__()
        self.cylinder_partition = cylinder_partition
        self.cylinder_range = cylinder_range
        
    def forward(self, points):
        res_points, res_coors = CylinderVoxelization(points, self.cylinder_partition, self.cylinder_range)
        return res_points, res_coors


class ScatterIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                coors,
                num_act_in):
        pts_voxel_maps = coors.new_zeros(coors.shape[0], dtype = torch.int32)
        voxel_coors = coors.new_zeros((num_act_in, 4), dtype = torch.int32)
        voxel_pts_counts = coors.new_zeros(num_act_in, dtype = torch.int32)
        return pts_voxel_maps, voxel_coors, voxel_pts_counts

    @staticmethod
    def symbolic(g,
                 coors,
                 num_act_in):
        data = np.array([num_act_in], dtype=np.int32).tobytes()
        return g.op('TRT_PluginV2', coors, name_s=b'ScatterIndex', data_s=data, namespace_s=b'', version_s=b'1.0',
                    outputs=3)

scatter_index = ScatterIndex.apply


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                pts_feats,
                pts_voxel_maps,
                voxel_pts_counts,
                reduce_op):
        voxel_feats = pts_feats.new_zeros((voxel_pts_counts.shape[0], pts_feats.shape[-1]), dtype = torch.float32)
        return voxel_feats

    @staticmethod
    def symbolic(g,
                 pts_feats,
                 pts_voxel_maps,
                 voxel_pts_counts,
                 reduce_op):
        data = reduce_op.encode()
        return g.op('TRT_PluginV2', pts_feats, pts_voxel_maps, voxel_pts_counts, name_s=b'_Reduce', data_s=data, namespace_s=b'', version_s=b'1.0',
                    outputs=1)

reduce = _Reduce.apply


class MapBack(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                voxel_feats,
                pts_voxel_maps):
        pts_feats = voxel_feats.new_zeros((pts_voxel_maps.shape[0], voxel_feats.shape[-1]), dtype = torch.float32)
        return pts_feats

    @staticmethod
    def symbolic(g,
                 voxel_feats,
                 pts_voxel_maps):
        data = b''
        return g.op('TRT_PluginV2', voxel_feats, pts_voxel_maps, name_s=b'MapBack', data_s=data, namespace_s=b'', version_s=b'1.0',
                    outputs=1)

mapback = MapBack.apply


class SpConvIdx3d(torch.autograd.Function):
    """
  /**
   * IO Part:
   *    Input:
   *        0: inCoors              [int32] [mMaxNumActIn, NDim + 1]
   *        1: numActIn             [int32] [1]
   *        2: inSpatialShape       [void]  [B, 0, Z, Y, X]  // dynamic shape
   *    Output:
   *        0: index                [int32] [3, kVol * mMaxNumActIn]
   *        1: (numBuf, bufSegLen)  [int32] [1 + kVol]
   *        2: outCoors             [int32] [mMaxNumActOut, NDim + 1]
   *        3: numActOut            [int32] [1]
   *        4: outSpatialShape      [void]  [B, 0, Z, Y, X]  // dynamic shape
   * */
    """

    @staticmethod
    def forward(ctx,
                in_coors: torch.Tensor,
                num_act_in: torch.Tensor,
                in_spatial_shape: torch.Tensor,
                kernel_size,
                stride,
                padding,
                dilation,
                out_padding,
                max_num_act_out,
                subm,
                transpose):
        max_num_act_in = in_coors.shape[0]
        kvol = int(np.prod(kernel_size))
        index = in_coors.new_zeros((3, kvol * max_num_act_in))
        index_buf_len = in_coors.new_zeros((1 + kvol,))
        if subm:
            return index, index_buf_len
        out_coors = in_coors.new_zeros((max_num_act_out, in_coors.shape[-1]))
        num_act_out = in_coors.new_zeros((1,))
        out_spatial_shape = list(in_spatial_shape.shape)
        if transpose:
            for i in range(3):
                out_spatial_shape[i + 2] = (
                        (out_spatial_shape[i + 2] - 1) * stride[i] - 2 * padding[i] + (kernel_size[i] - 1) *
                        dilation[i] + out_padding[i] + 1)

        else:
            for i in range(3):
                out_spatial_shape[i + 2] = (
                        (out_spatial_shape[i + 2] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[
                    i] + 1)

        out_spatial_shape = in_spatial_shape.new_zeros(out_spatial_shape)
        return index, index_buf_len, out_coors, num_act_out, out_spatial_shape

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 in_coors: torch._C.Value,
                 num_act_in: torch._C.Value,
                 in_spatial_shape: torch._C.Value,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 out_padding,
                 max_num_act_out,
                 subm,
                 transpose):
        data = np.array(kernel_size, dtype=np.int32).tobytes()
        data += np.array(stride, dtype=np.int32).tobytes()
        data += np.array(padding, dtype=np.int32).tobytes()
        data += np.array(dilation, dtype=np.int32).tobytes()
        data += np.array(out_padding, dtype=np.int32).tobytes()
        data += np.array([max_num_act_out], dtype=np.int32).tobytes()
        data += np.array([subm], dtype=np.bool).tobytes()
        data += np.array([transpose], dtype=np.bool).tobytes()
        return g.op('TRT_PluginV2',
                    in_coors, num_act_in, in_spatial_shape,
                    name_s=b"SpConvIdx3d", data_s=data, namespace_s=b'', version_s=b'1.0', outputs=2 if subm else 5)


spconv_index = SpConvIdx3d.apply


class SpConvMM(torch.autograd.Function):
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
                kernel_size,
                in_channels,
                out_channels,
                max_num_act_out,
                subm,
                inverse,
                weight,
                bias=None):
        out_feats = in_feats.new_zeros((max_num_act_out, out_channels))
        return out_feats

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 in_feats: torch._C.Value,
                 num_act_in: torch._C.Value,
                 num_act_out: torch._C.Value,
                 index: torch._C.Value,
                 index_buf_len: torch._C.Value,
                 kernel_size,
                 in_channels,
                 out_channels,
                 max_num_act_out,
                 subm,
                 inverse,
                 weight,
                 bias=None):
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
                    name_s=b"SpConvMM", data_s=data, namespace_s=b'', version_s=b'1.0', outputs=1)


spconv_mm = SpConvMM.apply


class InferSpConvModule(torch.nn.Module):
    def __init__(self, conv, bn, act, index_name, max_num_act_out):
        super(InferSpConvModule, self).__init__()
        if bn :
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
            index_dict[self.index_name] = indices
        index, index_buf_len = indices[:2]
        if self.conv.subm:
            out_coors, num_act_out, out_spatial_shape = in_coors, num_act_in, in_spatial_shape
        else:
            out_coors, num_act_out, out_spatial_shape = indices[2:]
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
            self.conv.transposed,
            self.weight,
            self.bias)
        if self.act:
            out_feats = self.act(out_feats)
        return [out_feats, out_coors, num_act_out, out_spatial_shape, index_dict]