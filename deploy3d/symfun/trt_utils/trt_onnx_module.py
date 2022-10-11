import torch
import tensorrt as trt
import glob
import ctypes
import pathlib
import hashlib
import numpy as np
import yaml
import os.path as osp
import os
import pickle
from urllib.request import urlopen


class TRTOnnxModule:
    libraries = []
    logger = None
    cache = '~/.cache/deploy3d'

    @staticmethod
    def _map_dtype(dtype):
        return {
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BOOL: torch.bool,
        }[dtype]

    def _prepare_io(self):
        assert self.context.all_shape_inputs_specified, 'not all dynamic binding shapes specified!'
        profile_idx = self.context.active_optimization_profile
        num_profiles = self.engine.num_optimization_profiles
        num_bindings_per_profile = self.engine.num_bindings // num_profiles
        if self.bindings is None:
            self.bindings = [0 for _ in range(
                num_bindings_per_profile * num_profiles)]
        if self.bindings_tensor is None:
            self.bindings_tensor = [None for _ in range(
                num_bindings_per_profile * num_profiles)]
        if self.active_bindings is None:
            self.active_bindings = dict()
        if self.mem_holder is None:
            self.mem_holder = [None for _ in range(
                num_bindings_per_profile * num_profiles)]
        for _io_idx in range(num_bindings_per_profile):
            io_idx = _io_idx + profile_idx * num_bindings_per_profile
            io_name = self.context.engine.get_binding_name(_io_idx)
            io_shape = torch.Size(self.context.get_binding_shape(io_idx))
            io_dtype = TRTOnnxModule._map_dtype(
                self.context.engine.get_binding_dtype(io_idx))
            if self.bindings[io_idx] == 0 or self.bindings_tensor[io_idx].shape != io_shape or self.bindings_tensor[
                    io_idx].dtype != io_dtype:  # not malloc yet or need re-allocate
                if torch.Size(io_shape).numel() == 0:
                    tensor = torch.empty([1], dtype=io_dtype, device='cuda')
                    self.mem_holder[io_idx] = tensor
                    self.bindings_tensor[io_idx] = tensor[:0].view(io_shape)
                    self.bindings[io_idx] = tensor.data_ptr()
                else:
                    tensor = torch.empty(
                        io_shape, dtype=io_dtype, device='cuda')
                    self.mem_holder[io_idx] = tensor
                    self.bindings_tensor[io_idx] = tensor
                    self.bindings[io_idx] = tensor.data_ptr()
            self.active_bindings[io_name] = self.bindings_tensor[io_idx]

    def _load_libraries(self):
        if len(TRTOnnxModule.libraries) == 0:
            libs_path = pathlib.Path(__file__).parents[2].joinpath('libs/*.so')
            libs = glob.glob(libs_path.as_posix())
            for lib in libs:
                TRTOnnxModule.libraries.append(ctypes.CDLL(lib))
            trt.init_libnvinfer_plugins(self._logger(), '')

    def _logger(self):
        if TRTOnnxModule.logger is None:
            TRTOnnxModule.logger = trt.Logger(trt.Logger.VERBOSE)
        return TRTOnnxModule.logger

    def build_engine(self, onnx_folder):
        if onnx_folder is None:
            model_onnx = []
            model_cfg = []
        else:
            model_onnx = glob.glob(osp.join(onnx_folder, '*.onnx'))
            model_cfg = glob.glob(osp.join(onnx_folder, '*.cfg'))

        if len(model_onnx) > 0:
            model_onnx = model_onnx[0]
            with open(model_onnx, 'rb') as f:
                model_onnx_bin = f.read()
            model_onnx_hash = hashlib.md5(model_onnx_bin).hexdigest()
        else:
            assert hasattr(self, 'model'), (
                'if `model.onnx` is not present, the model class should have '
                '`model` attributes, which is the url path to the model on '
                'intranet. Otherwise we have no idea what the model is.')
            model_onnx_hash = hashlib.md5(
                self.model.encode('utf-8')).hexdigest()
            model_onnx = osp.expanduser(osp.join(TRTOnnxModule.cache,
                                                 model_onnx_hash + '.onnx'))
            if not osp.exists(model_onnx):
                resp = urlopen(self.model)
                assert resp.getcode() == 200
                model_onnx_bin = resp.read()
                os.makedirs(osp.expanduser(
                    osp.join(TRTOnnxModule.cache)), exist_ok=True)
                with open(model_onnx, 'wb') as f:
                    f.write(model_onnx_bin)

        if len(model_cfg) > 0:
            model_cfg = model_cfg[0]
            with open(model_cfg, 'rb') as f:
                model_cfg_bin = f.read()
        else:
            assert hasattr(self, 'optimization_profiles'), (
                'if `model.cfg` is not present, the model class should have '
                '`optimization_profiles` attributes, otherwise we have no '
                'idea of the input dimensions so that we cannot optimize the '
                'model.')
            model_cfg = self.optimization_profiles
            model_cfg_bin = pickle.dumps(self.optimization_profiles)
        model_cfg_hash = hashlib.md5(model_cfg_bin).hexdigest()

        engine_name = hashlib.md5(
            (model_onnx_hash + model_cfg_hash).encode('utf-8')).hexdigest()
        engine_file = osp.expanduser(
            osp.join(TRTOnnxModule.cache, engine_name + '.engine'))

        if not osp.exists(engine_file):
            with trt.Builder(self._logger()) as builder, builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(
                    network, self._logger()) as parser:
                assert parser.parse_from_file(model_onnx), parser.get_error(0)
                config = builder.create_builder_config()
                config.max_workspace_size = 1 << 32  # 4GB
                config.flags = (1 << int(trt.BuilderFlag.FP16))

                if isinstance(model_cfg, str):
                    with open(model_cfg) as f:
                        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

                    input_opt = model_cfg['size']
                    input_min = model_cfg.get('size_min', input_opt)
                    input_max = model_cfg.get('size_max', input_opt)
                    assert len(input_opt) == len(input_min) == len(input_max)
                    for opt, min, max in zip(input_opt, input_min, input_max):
                        assert len(opt) == len(min) == len(max)
                        profile = builder.create_optimization_profile()
                        for in_idx, (in_opt, in_min, in_max) in enumerate(zip(opt, min, max)):
                            in_name = network.get_input(in_idx).name
                            profile.set_shape(in_name, in_min, in_opt, in_max)
                        config.add_optimization_profile(profile)
                else:
                    for optim_prof in model_cfg:
                        profile = builder.create_optimization_profile()
                        for in_name, in_shapes in optim_prof.items():
                            profile.set_shape(
                                in_name, in_shapes['min'], in_shapes['opt'], in_shapes['max'])
                        config.add_optimization_profile(profile)

                serialized_engine = builder.build_serialized_network(
                    network, config)
                assert serialized_engine, "cannot serialize engine!"
                os.makedirs(osp.expanduser(
                    osp.join(TRTOnnxModule.cache)), exist_ok=True)
                with open(engine_file, 'wb') as f:
                    f.write(serialized_engine)

        with open(engine_file, 'rb') as f:
            serialized_engine = f.read()

        self.runtime = trt.Runtime(self._logger())
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)

    def set_input_shape(self, **in_shapes):
        if len(in_shapes) == 0:
            in_shapes = self.in_shapes
            assert len(in_shapes) > 0

        if self.stream is None:
            self.stream = torch.cuda.current_stream()
        assert self.engine, 'engine not build!'
        if self.context is None:
            self.context = self.engine.create_execution_context()
        assert self.context, 'cannot create context!'
        num_profiles = self.engine.num_optimization_profiles
        num_bindings_per_profile = self.engine.num_bindings // num_profiles

        profiles_match_st = []
        for profile_idx in range(num_profiles):
            profile_match_st = []
            for in_name, in_shape in in_shapes.items():
                in_min, in_opt, in_max = self.engine.get_profile_shape(
                    profile_idx, in_name)
                if tuple(in_shape) == tuple(in_opt):
                    profile_match_st.append(2)  # optimum
                    continue
                elif len(in_shape) == len(in_opt) and all([a <= b <= c for a, b, c in zip(in_min, in_shape, in_max)]):
                    profile_match_st.append(1)  # valid
                    continue
                else:
                    profile_match_st.append(0)  # invalid
            profile_match_st = min(profile_match_st)
            if profile_match_st == 2:
                self.context.set_optimization_profile_async(
                    profile_idx, self.stream.cuda_stream)
                for in_name, in_shape in in_shapes.items():
                    in_idx = self.engine.get_binding_index(
                        in_name) + profile_idx * num_bindings_per_profile
                    self.context.set_binding_shape(in_idx, in_shape)
                assert self.context.all_binding_shapes_specified, 'not all dynamic binding shapes specified!'
                self._prepare_io()
                return
            profiles_match_st.append(profile_match_st)
        profile_idx = np.argmin(profiles_match_st)
        profiles_match_st = profiles_match_st[profile_idx]
        if profiles_match_st == 0:
            raise RuntimeError('dynamic input shape invalid!')
        self.context.set_optimization_profile_async(
            profile_idx, self.stream.cuda_stream)
        for in_name, in_shape in in_shapes.items():
            in_idx = self.engine.get_binding_index(
                in_name) + profile_idx * num_bindings_per_profile
            self.context.set_binding_shape(in_idx, in_shape)
        assert self.context.all_binding_shapes_specified, 'not all dynamic binding shapes specified!'
        self._prepare_io()

    def __init__(self, onnx_folder=None):
        self._load_libraries()
        self.runtime, self.engine, self.context = None, None, None
        self.stream = None
        self.bindings, self.bindings_tensor, self.mem_holder, self.active_bindings = None, None, None, None
        self.build_engine(onnx_folder)
        self.set_input_shape()

    def preprocess(self, *args, **kwargs):
        raise NotImplemented

    def postprocess(self, *args, **kwargs):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        stream = torch.cuda.current_stream()
        if stream.cuda_stream != self.stream.cuda_stream:
            self.stream.synchronize()
            self.stream = stream
        self.preprocess(*args, **kwargs)
        self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
        return self.postprocess(*args, **kwargs)
