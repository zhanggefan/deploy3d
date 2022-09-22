import tensorrt as trt
import torch.onnx
import glob
import ctypes
import io, os
import pathlib
import numpy as np
from functools import partial


class TRTPluginModule:
    libraries = []
    logger = None

    def __init__(self, onnx_file):
        self.sensor_signals = ['x', 'y', 'z', 'intensity']
        self.sensors = [0, 1, 2]
        self.point_radius_range = [0, 76.8]

        cls_res = torch.zeros((1, 1536), dtype=torch.int32)
        score = torch.zeros((1, 1536), dtype=torch.float32)
        bboxes = torch.zeros((1, 1536, 7), dtype=torch.float32)
        self.outputs = [cls_res, score, bboxes]
        
        self.context = partial(TRTPluginModule.create_context, onnx_file)()
        
        self.infer_onnx = partial(TRTPluginModule.forward, context=self.context, 
                                    outputs=self.outputs)
    
    @classmethod
    def _logger(cls):
        if cls.logger is None:
            cls.logger = trt.Logger(trt.Logger.VERBOSE)
        return cls.logger

    @classmethod
    def _load_libraries(cls):
        if len(cls.libraries) == 0:
            libs_path = pathlib.Path(__file__).parents[2].joinpath('deploy3d/libs/*.so')
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
    def create_context(cls, onnx_file):
        builder = trt.Builder(cls._logger())
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, cls._logger())
        # with open(onnx_file, 'rb') as f: 
        #     model = f.read()
        # assert parser.parse(model), parser.get_error(0)
        assert parser.parse_from_file(onnx_file), parser.get_error(0)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 32  # 4GB
        config.flags = (1 << int(trt.BuilderFlag.FP16))

        # profile = builder.create_optimization_profile()
        # profile.set_shape("batch_point_feats", (480000, 4), (480000, 4), (480000, 4))
        # profile.set_shape("batch_indices", (480000,), (480000,), (480000,))
        # profile.set_shape("in_spatial_shape", (1, 0, 41, 1536, 1536), (1, 0, 41, 1536, 1536), (1, 0, 41, 1536, 1536))
        # config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        assert serialized_engine, "cannot serialize engine!"
        runtime = trt.Runtime(cls._logger())
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        assert context, "failed to make execution context!"
        return context
    
    @classmethod
    def forward(cls, input_tensors, context, outputs):
        inputs = cls._malloc(input_tensors)
        outputs = cls._malloc(outputs)
        bindings, mem_holder = cls._to_bindings(context, inputs, outputs)
        
        context.execute_v2(bindings)
        return outputs

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

    def inference(self, pts_file):
        points = self.load_points(pts_file)
        points = points.reshape(-1, points.shape[-1])

        # create input tensors, can be placed in init part
        batch_point_feats = torch.from_numpy(points, dtype=torch.float32) # (n, f)
        batch_indices = torch.zeros(batch_point_feats.shape[0], dtype=torch.int32) # (n,)
        voxel_config = torch.tensor([-76.8, -76.8, -2, 0.1, 0.1, 0.15]) # (6,) [xmin, ymin, zmin, dx, dy, dz]
        in_spatial_shape = torch.empty((1, 0, 41, 1536, 1536), dtype=torch.int32) # (1, 0, z, y, x)
        input_tensors = [batch_point_feats, batch_indices, voxel_config, in_spatial_shape]
        
        outputs = self.infer_onnx(input_tensors)

        # post process
        
        return outputs

    
if __name__ == '__main__':
    # input_path = '/ssd0/zy_dataset/jinghu/dataset/3/pc_out'
    # result_path = '/ssd0/zy_dataset/jinghu/dataset/3/pc_bboxes'
    onnx_file = './models/det3d_ruby/yolox3d_voxel_ruby_cowadataset_1f.onnx'
    
    # pts_files = glob.glob(os.path.join(input_path, '*.npy'))
    pts_files = []
    trt_module = TRTPluginModule(onnx_file)
    for pts_file in pts_files:
        trt_module.inference(pts_file)