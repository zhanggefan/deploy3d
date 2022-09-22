import tensorrt as trt
import torch.onnx
import glob
import ctypes
import io
import pathlib


class ModuleOfFunction(torch.nn.Module):
    def __init__(self, torch_fun, *config_args):
        super(ModuleOfFunction, self).__init__()
        self.torch_fun = torch_fun
        self.config_args = config_args

    def forward(self, *args):
        with torch.no_grad():
            return self.torch_fun.apply(*args, *self.config_args)


class Profiler(trt.IProfiler):
    """
    Example Implimentation of a Profiler
    Is identical to the Profiler class in trt.infer so it is possible
    to just use that instead of implementing this if further
    functionality is not needed
    """
    def __init__(self, timing_iter):
        trt.IProfiler.__init__(self)
        self.timing_iterations = timing_iter
        self.profile = []

    def report_layer_time(self, layerName, ms):
        record = next((r for r in self.profile if r[0] == layerName), (None, None))
        if record == (None, None):
            self.profile.append((layerName, ms))
        else:
            self.profile[self.profile.index(record)] = (record[0], record[1] + ms)

    def print_layer_times(self):
        totalTime = 0
        for i in range(len(self.profile)):
            print("{:40.40} {:4.3f}ms".format(self.profile[i][0], self.profile[i][1] / self.timing_iterations))
            totalTime += self.profile[i][1]
        print("Time over all layers: {:4.2f} ms per iteration".format(totalTime / self.timing_iterations))


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
            libs_path = pathlib.Path(__file__).parents[2].joinpath('libs/*.so')
            libs = glob.glob(libs_path.as_posix())
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
            binding_shape = context.engine.get_binding_shape(idx)
            in_out, in_out_idx = binding_name.split('_')
            in_out_idx = int(in_out_idx)
            t = inputs[in_out_idx] if in_out == 'in' else outputs[in_out_idx]
            assert tuple(t.shape) == tuple(
                binding_shape), (f'wrong io shape of '
                                 f'{binding_name} which should be {binding_shape}')
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
        is_single_tensor = not isinstance(outputs, (list, tuple))
        num_outputs = 1 if is_single_tensor else len(outputs)
        torch.onnx.export(module, tuple(input_tensors), f,
                          enable_onnx_checker=False,
                          input_names=[f'in_{idx}' for idx in range(
                              len(input_tensors))],
                          output_names=[f'out_{idx}' for idx in range(num_outputs)],
                          opset_version=9)
        builder = trt.Builder(cls._logger())
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, cls._logger())
        assert parser.parse(f.getvalue()), parser.get_error(0)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 32  # 4GB
        config.flags = (1 << int(trt.BuilderFlag.FP16))
        serialized_engine = builder.build_serialized_network(network, config)
        assert serialized_engine, "cannot serialize engine!"
        runtime = trt.Runtime(cls._logger())
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        assert context, "failed to make execution context!"
        # time_infer
        g_prof = Profiler(1)
        context.profiler = g_prof
        
        inputs = cls._malloc(input_tensors)
        outputs = cls._malloc(outputs)
        bindings, mem_holder = cls._to_bindings(context, inputs, outputs)
        context.execute_v2(bindings)
        g_prof.print_layer_times()  # print infer time
        if is_single_tensor:
            return outputs[0]
        else:
            return outputs
