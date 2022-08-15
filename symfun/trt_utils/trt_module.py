import tensorrt as trt
import torch.onnx
import glob
import ctypes
import io


class ModuleOfFunction(torch.nn.Module):
    def __init__(self, torch_fun, *config_args):
        super(ModuleOfFunction, self).__init__()
        self.torch_fun = torch_fun
        self.config_args = config_args

    def forward(self, *args):
        with torch.no_grad():
            return self.torch_fun.apply(*args, *self.config_args)


class TRTPluginModule:
    libraries = []
    logger = None

    @classmethod
    def _logger(cls):
        if cls.logger is None:
            cls.logger = trt.Logger(trt.Logger.VERBOSE)
        return cls.logger

    @classmethod
    def _load_libraries(cls):
        if len(cls.libraries) == 0:
            libs = glob.glob('libs/*.so')
            for lib in libs:
                cls.libraries.append(ctypes.CDLL(lib))
            trt.init_libnvinfer_plugins(cls._logger(), '')

    @staticmethod
    def _map_dtype(dtype):
        return {
            'torch.int32': trt.tensorrt.int32,
            'torch.int8': trt.tensorrt.int8,
            'torch.float32': trt.tensorrt.float32,
            'torch.float16': trt.tensorrt.float16,
            'torch.bool': trt.tensorrt.bool,
        }[str(dtype)]

    @staticmethod
    def _malloc(tensors):
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        return [t.detach().clone().cuda() for t in tensors]

    @staticmethod
    def _to_bindings(context, inputs, outputs):
        bindings = []
        mem_holder = []
        for idx in range(context.engine.num_bindings):
            binding_name = context.engine.get_binding_name(idx)
            in_out, in_out_idx = binding_name.split('_')
            in_out_idx = int(in_out_idx)
            t = inputs[in_out_idx] if in_out == 'in' else outputs[in_out_idx]
            mem_holder.append(t if t.numel() else t.new_empty([1]))
            bindings.append(mem_holder[-1].data_ptr())
        return bindings, mem_holder

    @classmethod
    def forward(cls, module, input_tensors, configs=[]):
        cls._load_libraries()
        if not isinstance(module, torch.nn.Module):
            module = ModuleOfFunction(module, *configs)
        f = io.BytesIO()
        outputs = module(*input_tensors)
        torch.onnx.export(module, tuple(input_tensors), f,
                          enable_onnx_checker=False,
                          input_names=[f'in_{idx}' for idx in range(
                              len(input_tensors))],
                          output_names=[f'out_{idx}' for idx in range(len(outputs))])
        builder = trt.Builder(cls._logger())
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, cls._logger())
        assert parser.parse(f.getvalue()), parser.get_error(0)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 32  # 4GB
        serialized_engine = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(cls._logger())
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        assert context, "failed to make execution context!"

        is_single_tensor = not isinstance(outputs, (list, tuple))
        inputs = cls._malloc(input_tensors)
        outputs = cls._malloc(outputs)
        bindings, mem_holder = cls._to_bindings(context, inputs, outputs)
        context.execute_v2(bindings)
        if is_single_tensor:
            return outputs[0]
        else:
            return outputs
