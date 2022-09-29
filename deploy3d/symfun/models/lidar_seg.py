from ..trt_utils import TRTOnnxModule
import torch
import numpy as np
import io


class LidarSegRubyOuster(TRTOnnxModule):
    optimization_profiles = [
        {'batch_point_feats': {'opt': (480000, 4),
                               'min': (480000, 4),
                               'max': (480000, 4)},
         'batch_indices': {'opt': (480000,),
                           'min': (480000,),
                           'max': (480000,)},
         'voxel_config': {'opt': (6,),
                          'min': (6,),
                          'max': (6,)},
         'in_spatial_shape': {'opt': (1, 0, 480, 360, 32),
                              'min': (1, 0, 480, 360, 32),
                              'max': (1, 0, 480, 360, 32)}}]
    in_shapes = {'batch_point_feats': (480000, 4),
                 'batch_indices': (480000,),
                 'cylinder_config': (6,),
                 'in_spatial_shape': (1, 0, 480, 360, 32)}
    sensors = [[0, 1, 2]]
    cylinder_config = [-2, -np.pi, 0, 4, np.pi, 50]

    def __init__(self, onnx_folder):
        super(LidarSegRubyOuster, self).__init__(onnx_folder)
        self.active_bindings['cylinder_config'][:] = torch.tensor(
            self.cylinder_config)

    def _read_pts(self, points):
        if isinstance(points, str) and points.endswith('.npy'):
            return np.load(points)
        elif isinstance(points, (bytes, bytearray)):
            return np.load(io.BytesIO(points))
        elif isinstance(points, np.ndarray):
            return points
        else:
            raise RuntimeError('unsupported input type!')

    def _npy2array(self, points):
        return np.stack([points['x'].astype(np.float32),
                         points['y'].astype(np.float32),
                         points['z'].astype(np.float32),
                         points['intensity'].astype(np.float32) / 255], axis=-1)
    
    def radius_range_filter(self, raw_points, min_range=1.5, max_range=50):
        points = np.stack([raw_points['x'], raw_points['y'], raw_points['z']], axis=-1)
        distance = np.linalg.norm(points[:, :2], ord=2, axis=-1)
        mask = (distance > min_range) & (distance < max_range)
        return mask, raw_points[mask]
    
    def preprocess(self, points):
        points = self._read_pts(points)
        self.mask, points = self.radius_range_filter(points)
        pts_sensor = points['sensor']
        batch_indices = np.full([pts_sensor.shape[0]], -1)
        for batch_idx, batch_sensors in enumerate(self.sensors):
            for sensor in batch_sensors:
                batch_indices[pts_sensor == sensor] = batch_idx
        sensor_mask = batch_indices >= 0
        points = points[sensor_mask]
        batch_indices = batch_indices[sensor_mask]
        points = self._npy2array(points)
        num_points = points.shape[0]
        if num_points > self.active_bindings['batch_point_feats'].shape[0]:
            self._logger().WARNING('discard input points because the number of input is too large!')
            num_points = self.active_bindings['batch_point_feats'].shape[0]
        self.active_bindings['batch_point_feats'][:num_points] = torch.from_numpy(points)[:num_points].to(
            dtype=self.active_bindings['batch_point_feats'].dtype)
        self.active_bindings['batch_indices'][:num_points] = torch.from_numpy(batch_indices)[:num_points].to(
            dtype=self.active_bindings['batch_point_feats'].dtype)
        self.active_bindings['batch_indices'][num_points:] = -1

    def postprocess(self, points):
        labels = self.active_bindings['batch_point_labels']
        labels = labels.cpu().numpy()
        return dict(labels=labels, mask=self.mask)
