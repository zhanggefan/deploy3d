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
        return [t.detach().clone().cuda() for t in tensors]

    @staticmethod
    def _to_bindings(tensors):
        bindings = []
        mem_holder = []
        for t in tensors:
            mem_holder.append(t if t.numel() else t.new_empty([1]))
            bindings.append(mem_holder[-1].data_ptr())
        return bindings, mem_holder

    @classmethod
    def forward(cls, module, input_tensors, configs=[]):
        cls._load_libraries()
        if issubclass(module, torch.autograd.Function):
            module = ModuleOfFunction(module, *configs)
        f = io.BytesIO()
        torch.onnx.export(module, tuple(input_tensors), f,
                          enable_onnx_checker=False)
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

        outputs = module(*input_tensors)
        inputs = cls._malloc(input_tensors)
        outputs = cls._malloc(outputs)
        bindings, mem_holder = cls._to_bindings(inputs + outputs)
        context.execute_v2(bindings)
        return outputs
