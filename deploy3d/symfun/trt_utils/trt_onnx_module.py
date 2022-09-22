from cmath import e
import tensorrt as trt
import torch.onnx
import glob
import ctypes
import io, os
import pathlib
import numpy as np
from functools import partial


class TRTOnnxModule:
    libraries = []
    logger = None

    def __init__(self, onnx_file):
        self.engine = None
        self.onnx_file = onnx_file
        
        self.sensor_signals = ['x', 'y', 'z', 'intensity']
        self.sensors = [0, 1, 2]
        self.point_radius_range = [0, 76.8]
        self.max_pts = 480000
        
        cls_id = torch.zeros((1, 1536), dtype=torch.int32)
        score = torch.zeros((1, 1536), dtype=torch.float32)
        bboxes = torch.zeros((1, 1536, 7), dtype=torch.float32)
        self.outputs = [cls_id, score, bboxes]
        
        self._load_libraries()
    
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
        engine_file = os.path.splitext(onnx_file)[0] + '.engine'
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
            profile.set_shape("batch_point_feats", (480000, 4), (480000, 4), (480000, 4))
            profile.set_shape("batch_indices", (480000,), (480000,), (480000,))
            profile.set_shape("voxel_config", (6,), (6,), (6,))
            profile.set_shape("in_spatial_shape", (1, 0, 41, 1536, 1536), (1, 0, 41, 1536, 1536), (2, 0, 41, 1536, 1536))
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
        context.set_binding_shape(0, (480000, 4))
        context.set_binding_shape(1, (480000,))
        context.set_binding_shape(2, (6,))
        context.set_binding_shape(3, (1, 0, 41, 1536, 1536))
        
        return context

    def radius_range_filter(self, points):
        min_range, max_range = self.point_radius_range
        distance = np.linalg.norm(points[:, :2], ord=2, axis=-1)
        range_mask = (distance > min_range) & (distance < max_range)
        points = points[range_mask]
        return points

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
        
        pts = self.radius_range_filter(pts)
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
        voxel_config = torch.tensor([-76.8, -76.8, -2, 0.1, 0.1, 0.15]) # (6,) [xmin, ymin, zmin, dx, dy, dz]
        in_spatial_shape = torch.empty((1, 0, 41, 1536, 1536), dtype=torch.int32) # (1, 0, z, y, x)
        input_tensors = [batch_point_feats, batch_indices, voxel_config, in_spatial_shape]
        
        # 3. run infer
        inputs = self._malloc(input_tensors)
        outputs = self._malloc(self.outputs)
        bindings, mem_holder = self._to_bindings(self.context, inputs, outputs)
        
        self.context.execute_v2(bindings)
        
        # post process
        _, scores, bboxes = outputs
        
        mask = (scores.squeeze() > 0)
        bboxes = bboxes.squeeze()[mask]
        return bboxes.cpu().numpy()