import torch


class NMS3d(torch.autograd.Function):
    """
    /**
     * IO Part:
     *    Input:
     *        0: inScores             [float]     [b, topK]
     *        1: boxes                [float]     [b, topK, 7 + a]
     *    Output:
     *        0: outScores            [float]     [b, topK]
     * */
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor,
                boxes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(scores)

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 scores: torch._C.Value,
                 boxes: torch._C.Value) -> torch._C.Value:
        return g.op('TRT_PluginV2', scores, boxes, name_s=b'NMS3d', data_s=b'',
                    namespace_s=b'', version_s=b'2.0', outputs=1)


nms3d = NMS3d.apply
