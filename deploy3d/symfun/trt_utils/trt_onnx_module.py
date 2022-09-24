from cmath import e
import tensorrt as trt
import torch.onnx
import glob
import ctypes
import io, os
import pathlib
import hashlib
import numpy as np
import configparser


class TRTOnnxModule:
    libraries = []
    logger = None
    input_names = ('batch_point_feats',
                   'batch_indices',
                   'voxel_config',
                   'in_spatial_shape')
    
    def __init__(self, onnx_file, config_path, yaml_file):
        self.engine = None
        self.onnx_file = onnx_file
        self.sensor_signals = ['x', 'y', 'z', 'intensity']
        
        config = self.read_cfg(config_path)
        self.input_sizes = self.get_input_size(config, self.input_names)
        
        self.max_pts = self.input_sizes['batch_indices']['size'][0]
        
        sp_size = self.input_sizes['in_spatial_shape']['size']
        self.voxel_config = torch.tensor([-51.2, -(sp_size[-2] / 20), -2, 0.1, 0.1, 0.15]) # (6,) [xmin, ymin, zmin, dx, dy, dz]
        self.in_spatial_shape = torch.empty(sp_size, dtype=torch.int32) # (1, 0, z, y, x)
        
        params = self.parse_yaml(yaml_file, onnx_file)
        self.sensors = params['sensors'][0]
        self.score_threshold = params['score_threshold']
        
        cls_id = torch.zeros((1, 1536), dtype=torch.int32)
        score = torch.zeros((1, 1536), dtype=torch.float32)
        bboxes = torch.zeros((1, 1536, 7), dtype=torch.float32)
        self.outputs = [cls_id, score, bboxes]
        
        self._load_libraries()
    
    def read_cfg(self, config_path):
        conf = configparser.ConfigParser()
        with open(config_path, 'r') as f:
            buf = f.read()
            buf = '[default]\n' + buf
        conf.read_string(buf)
            
        config = conf['default']
        return config
    
    def get_input_size(self, config, input_names):
        size = eval(config['size'])[0]
        size_min = eval(config['size_min'])[0]
        size_max = eval(config['size_max'])[0]
        
        input_sizes = {input_name: dict(size=tuple(size[idx]),
                                        size_min=tuple(size_min[idx]),
                                        size_max=tuple(size_max[idx])) 
                             for idx, input_name in enumerate(input_names)}
        
        value = input_sizes['in_spatial_shape']['size']
        input_sizes['in_spatial_shape']['size'] = tuple([1] + list(value[1:])) # for yolox3d onnx infer, bs always be 1
        return input_sizes
    
    def parse_yaml(self, yaml_file, onnx_file):
        import yaml
        from yaml.loader import SafeLoader

        with open(yaml_file) as f:
            data = yaml.load(f, Loader=SafeLoader)
        
        prefix = os.path.splitext(os.path.split(onnx_file)[-1])[0]
        if 'ruby' in prefix:
            key = 'lidardetruby'
        else:
            key = 'lidardetouster'
            self.sensor_signals.extend(['value'])
        params = data['LidarPerception'][key]
        return params
    
    def _logger(self):
        if self.logger is None:
            self.logger = trt.Logger(trt.Logger.VERBOSE)
        return self.logger

    def _load_libraries(self):
        if len(self.libraries) == 0:
            libs_path = pathlib.Path(__file__).parents[2].joinpath('libs/*.so')
            libs = glob.glob(libs_path.as_posix())
            for lib in libs:
                self.libraries.append(ctypes.CDLL(lib))
            trt.init_libnvinfer_plugins(self._logger(), '')

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
        in_out_dict = dict(batch_point_feats='in_0',
                           batch_indices='in_1',
                           voxel_config='in_2',
                           in_spatial_shape='in_3',
                           cls='out_0',
                           score='out_1',
                           box='out_2')
        
        for idx in range(context.engine.num_bindings):
            binding_name = context.engine.get_binding_name(idx)
            binding_shape = context.get_binding_shape(idx)
            
            in_out, in_out_idx = in_out_dict[binding_name].split('_')
            in_out_idx = int(in_out_idx)
            t = inputs[in_out_idx] if in_out == 'in' else outputs[in_out_idx]
            assert tuple(t.shape) == tuple(
                binding_shape), (f'wrong io shape of '
                                 f'{binding_name} which should be {binding_shape}')
            mem_holder.append(t if t.numel() else t.new_empty([1]))
            bindings.append(mem_holder[-1].data_ptr())
        return bindings, mem_holder

    def build_engine(self, onnx_file):
        cache_path = '/root/.cache'
        prefix = os.path.splitext(os.path.split(onnx_file)[-1])[0]
        engine_name = hashlib.md5(prefix.encode()).hexdigest()
        
        engine_file = os.path.join(cache_path, engine_name + '.engine')
        if not os.path.exists(engine_file):
            builder = trt.Builder(self._logger())
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self._logger())
            assert parser.parse_from_file(onnx_file), parser.get_error(0)
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 32  # 4GB
            config.flags = (1 << int(trt.BuilderFlag.FP16))
            
            profile = builder.create_optimization_profile()
            for name in self.input_names:
                in_size = self.input_sizes[name]
                profile.set_shape(name, in_size['size_min'], in_size['size_min'], in_size['size_max'])
            config.add_optimization_profile(profile)

            serialized_engine = builder.build_serialized_network(network, config)
            assert serialized_engine, "cannot serialize engine!"
            with open(engine_file, 'wb') as f:
                f.write(serialized_engine)
        
        with open(engine_file, 'rb') as f:
            serialized_engine = f.read()
        
        runtime = trt.Runtime(self._logger())
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine
    
    def create_context(self, engine):
        context = engine.create_execution_context()
        assert context, "failed to make execution context!"
        
        context.active_optimization_profile = 0
        for name in self.input_names:
            in_size = self.input_sizes[name]
            idx = engine.get_binding_index(name)
            context.set_binding_shape(idx, in_size['size'])
        
        return context

    def load_points(self, pts_file):
        pts = np.load(pts_file)
        sensor_mask = None
        if self.sensors is not None:
            sensor_mask = [pts['sensor'] == s for s in self.sensors]
            sensor_mask = np.stack(sensor_mask, axis=-1).any(axis=-1)
            pts = pts[sensor_mask]
        sig_vals = []
        for sig_name in self.sensor_signals:
            sig_val = pts[sig_name].astype(np.float32)
            if sig_name in ['intensity', 'value']:
                sig_val /= 255.0
            sig_vals.append(sig_val)
        pts = np.stack(sig_vals, axis=-1)
        
        return pts

    def forward(self, pts_file):
        # 1. build engine & create context
        if not self.engine:
            engine = self.build_engine(self.onnx_file)
            self.context = self.create_context(engine)
            self.engine = engine
        
        # 2. prepare inputs
        points = self.load_points(pts_file)
        points = points.reshape(-1, points.shape[-1])
        
        if points.shape[0] >= self.max_pts:
            points = points[0:self.max_pts, :]
        else:
            expanded_num = self.max_pts - points.shape[0]
            expanded_pts = np.zeros((expanded_num, points.shape[1]), dtype=np.float32)
            points = np.concatenate([points, expanded_pts], axis=0)
        
        assert points.shape[0] == self.max_pts
        
        batch_point_feats = torch.from_numpy(points) # (n, f)
        batch_indices = torch.zeros(batch_point_feats.shape[0], dtype=torch.int32) # (n,)
        voxel_config = self.voxel_config
        in_spatial_shape = self.in_spatial_shape
        input_tensors = [batch_point_feats, batch_indices, voxel_config, in_spatial_shape]
        
        # 3. run infer
        inputs = self._malloc(input_tensors)
        outputs = self._malloc(self.outputs)
        bindings, mem_holder = self._to_bindings(self.context, inputs, outputs)
        
        self.context.execute_v2(bindings)
        
        # 4. post process
        bboxes = self.post_process(outputs)
        return bboxes
    
    def post_process(self, outputs):
        cls_ids, scores, bboxes = outputs
        cls_ids, scores, bboxes = cls_ids.squeeze(), scores.squeeze(), bboxes.squeeze()
        cls_ids, scores, bboxes = cls_ids.cpu().numpy(), scores.cpu().numpy(), bboxes.cpu().numpy()
        
        labels = -1 * np.ones(cls_ids.shape[0], dtype=np.uint8)
        mask = np.zeros(cls_ids.shape[0], dtype=np.bool)
        for label_id, _ in enumerate(self.score_threshold):
            cls_mask = (cls_ids == label_id) & (scores >= self.score_threshold[label_id])
            labels[cls_mask] = label_id
            mask += cls_mask
        bboxes = bboxes[mask]
        labels = labels[mask]
        return np.concatenate([labels[:, None], bboxes], axis=-1)