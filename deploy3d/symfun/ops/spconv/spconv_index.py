from typing import Tuple, Union
import torch
import numpy as np


class SPConvIdx3d(torch.autograd.Function):
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
   *      Optional:
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
                transpose) -> Union[
        Tuple[
            torch.Tensor,
            torch.Tensor],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor]]:
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
                        (out_spatial_shape[i + 2] - 1) * stride[i] - 2 *
                        padding[i] + (kernel_size[i] - 1) *
                        dilation[i] + out_padding[i] + 1)

        else:
            for i in range(3):
                out_spatial_shape[i + 2] = (
                        (out_spatial_shape[i + 2] + 2 * padding[i] - dilation[
                            i] * (kernel_size[i] - 1) - 1) // stride[
                            i] + 1)

        out_spatial_shape = in_spatial_shape.new_zeros(out_spatial_shape)
        return index, index_buf_len, out_coors, num_act_out, out_spatial_shape

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 in_coors: torch._C.Value,
                 num_act_in: torch._C.Value,
                 in_spatial_shape: torch._C.Value,
                 kernel_size: Union[tuple, list],
                 stride,
                 padding,
                 dilation,
                 out_padding,
                 max_num_act_out,
                 subm,
                 transpose) -> Union[
        Tuple[
            torch._C.Value,
            torch._C.Value],
        Tuple[
            torch._C.Value,
            torch._C.Value,
            torch._C.Value,
            torch._C.Value,
            torch._C.Value]]:
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
                    name_s=b"SPConvIdx3d", data_s=data, namespace_s=b'',
                    version_s=b'2.0', outputs=2 if subm else 5)


spconv_index = SPConvIdx3d.apply
