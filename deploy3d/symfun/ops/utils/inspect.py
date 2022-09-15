import torch


class Inspect(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                tensor: torch.Tensor,
                name: str) -> torch.Tensor:
        return tensor

    @staticmethod
    def symbolic(g: torch._C.Graph,
                 tensor: torch._C.Value,
                 name: str) -> torch._C.Value:
        data = name.encode()
        return g.op('TRT_PluginV2', tensor, name_s=b'Inspect', data_s=data,
                    namespace_s=b'', version_s=b'2.0', outputs=1)


inspect = Inspect.apply
