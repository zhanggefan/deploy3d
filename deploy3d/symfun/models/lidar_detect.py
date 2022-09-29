from ..trt_utils import TRTOnnxModule
import torch
import numpy as np
import io
from scipy.spatial.transform import Rotation as R


class LidarDetRuby(TRTOnnxModule):
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
         'in_spatial_shape': {'opt': (1, 0, 41, 1536, 1536),
                              'min': (1, 0, 41, 1536, 1536),
                              'max': (1, 0, 41, 1536, 1536)}}]
    in_shapes = {'batch_point_feats': (480000, 4),
                 'batch_indices': (480000,),
                 'voxel_config': (6,),
                 'in_spatial_shape': (1, 0, 41, 1536, 1536)}
    sensors = [[0]]
    voxel_config = [-51.2, -76.8, -2.0, 0.1, 0.1, 0.15]
    classes = ['BUS', 'PEDESTRIAN', 'CAR', 'CYCLIST', 'TRICYCLE', 'ROADBLOCK']
    score_threshold = [0.7, 0.5, 0.7, 0.5, 0.5, 0.3]

    def __init__(self, onnx_folder):
        super(LidarDetRuby, self).__init__(onnx_folder)
        self.active_bindings['voxel_config'][:] = torch.tensor(
            self.voxel_config)

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

    def preprocess(self, points):
        points = self._read_pts(points)
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
        cls_ids, scores, bboxes = (
            self.active_bindings['cls'], self.active_bindings['score'], self.active_bindings['box'])
        cls_ids, scores, bboxes = cls_ids.squeeze(), scores.squeeze(), bboxes.squeeze()
        cls_ids, scores, bboxes = cls_ids.cpu().numpy(
        ), scores.cpu().numpy(), bboxes.cpu().numpy()

        out_labels = []
        out_scores = []
        out_bboxes = []
        for label_id, _ in enumerate(self.score_threshold):
            cls_mask = (cls_ids == label_id) & (
                scores >= self.score_threshold[label_id])
            out_scores.append(scores[cls_mask])
            out_bboxes.append(bboxes[cls_mask])
            out_labels.append([self.classes[label_id]
                              for _ in range(out_scores[-1].shape[0])])

        out_bboxes = np.concatenate(out_bboxes, axis=0)
        out_scores = np.concatenate(out_scores, axis=0)
        out_labels = sum(out_labels, [])

        return dict(labels_3d=out_labels, scores_3d=out_scores, boxes_3d=out_bboxes)


class LidarDetOuster(LidarDetRuby):
    optimization_profiles = [
        {'batch_point_feats': {'opt': (480000, 5),
                               'min': (480000, 5),
                               'max': (480000, 5)},
         'batch_indices': {'opt': (480000,),
                           'min': (480000,),
                           'max': (480000,)},
         'voxel_config': {'opt': (6,),
                          'min': (6,),
                          'max': (6,)},
         'in_spatial_shape': {'opt': (1, 0, 41, 1024, 1024),
                              'min': (1, 0, 41, 1024, 1024),
                              'max': (1, 0, 41, 1024, 1024)}}]
    in_shapes = {'batch_point_feats': (480000, 5),
                 'batch_indices': (480000,),
                 'voxel_config': (6,),
                 'in_spatial_shape': (1, 0, 41, 1024, 1024)}
    sensors = [[1, 2]]
    voxel_config = [-51.2, -51.2, -2.0, 0.1, 0.1, 0.15]
    classes = ['BUS', 'PEDESTRIAN', 'CAR', 'CYCLIST', 'TRICYCLE', 'ROADBLOCK']
    score_threshold = [0.7, 0.5, 0.7, 0.5, 0.5, 0.3]

    def _npy2array(self, points):
        return np.stack([points['x'].astype(np.float32),
                         points['y'].astype(np.float32),
                         points['z'].astype(np.float32),
                         points['intensity'].astype(np.float32) / 255,
                         points['value'].astype(np.float32) / 255], axis=-1)


class LidarDetRubyVel(LidarDetRuby):
    optimization_profiles = [
        {'batch_point_feats': {'opt': (480000, 5),
                               'min': (480000, 5),
                               'max': (480000, 5)},
         'batch_indices': {'opt': (480000,),
                           'min': (480000,),
                           'max': (480000,)},
         'voxel_config': {'opt': (6,),
                          'min': (6,),
                          'max': (6,)},
         'in_spatial_shape': {'opt': (1, 0, 41, 1536, 1536),
                              'min': (1, 0, 41, 1536, 1536),
                              'max': (1, 0, 41, 1536, 1536)}}]
    in_shapes = {'batch_point_feats': (480000, 5),
                 'batch_indices': (480000,),
                 'voxel_config': (6,),
                 'in_spatial_shape': (1, 0, 41, 1536, 1536)}
    sensors = [[0]]
    voxel_config = [-51.2, -76.8, -2.0, 0.1, 0.1, 0.15]
    classes = ['BUS', 'PEDESTRIAN', 'CAR', 'CYCLIST', 'TRICYCLE', 'ROADBLOCK']
    score_threshold = [0.7, 0.5, 0.7, 0.5, 0.5, 0.3]

    def _npy2array(self, points, ts, pose):
        xyz1 = np.stack([points['x'].astype(np.float32),
                         points['y'].astype(np.float32),
                         points['z'].astype(np.float32),
                         np.ones([points.shape[0]], dtype=np.float32)],
                        axis=-1)
        xyz1 = xyz1 @ pose.T
        return np.stack([xyz1[:, 0],
                         xyz1[:, 1],
                         xyz1[:, 2],
                         points['intensity'].astype(np.float32) / 255,
                         np.full([points.shape[0]], ts, dtype=np.float32)],
                        axis=-1)

    @staticmethod
    def get_ts(multiframe_metas):
        if isinstance(multiframe_metas[0]['timestamp'], dict):
            def _get_ts(x): return int(x['timestamp']['$numberLong'])/1e9
        else:
            def _get_ts(x): return x['timestamp']/1e9
        ts0 = _get_ts(multiframe_metas[0])
        return [ts0 - _get_ts(m) for m in multiframe_metas]

    @staticmethod
    def get_pose(multiframe_metas):
        poses = []
        for m in multiframe_metas:
            p = m['pose']
            mat = np.eye(4, dtype=float)
            mat[:3, :3] = R.from_euler(
                'ZYX',
                [p['yaw'], p['pitch'], p['roll']],
                degrees=False).as_matrix()
            mat[:3, 3] = np.array([p['x'], p['y'], p['z']], dtype=float)
            poses.append(mat)
        ret = [np.eye(4, dtype=np.float32)]
        ret += [(np.linalg.inv(poses[0]) @ p).astype(np.float32)
                for p in poses[1:]]
        return ret

    def preprocess(self, multiframe_points, multiframe_metas):
        tss = self.get_ts(multiframe_metas)
        poses = self.get_pose(multiframe_metas)

        mf_points = []
        mf_batch_indices = []

        for points, ts, pose in zip(multiframe_points, tss, poses):
            points = self._read_pts(points)
            pts_sensor = points['sensor']
            batch_indices = np.full([pts_sensor.shape[0]], -1)
            for batch_idx, batch_sensors in enumerate(self.sensors):
                for sensor in batch_sensors:
                    batch_indices[pts_sensor == sensor] = batch_idx
            sensor_mask = batch_indices >= 0
            points = points[sensor_mask]
            batch_indices = batch_indices[sensor_mask]
            mf_batch_indices.append(batch_indices)
            points = self._npy2array(points, ts, pose)
            mf_points.append(points)

        mf_points = np.concatenate(mf_points, axis=0)
        mf_batch_indices = np.concatenate(mf_batch_indices, axis=0)

        num_points = mf_points.shape[0]
        if num_points > self.active_bindings['batch_point_feats'].shape[0]:
            self._logger().WARNING('discard input points because the number of input is too large!')
            num_points = self.active_bindings['batch_point_feats'].shape[0]
        self.active_bindings['batch_point_feats'][:num_points] = torch.from_numpy(mf_points)[:num_points].to(
            dtype=self.active_bindings['batch_point_feats'].dtype)
        self.active_bindings['batch_indices'][:num_points] = torch.from_numpy(mf_batch_indices)[:num_points].to(
            dtype=self.active_bindings['batch_point_feats'].dtype)
        self.active_bindings['batch_indices'][num_points:] = -1

    def postprocess(self, multiframe_points, multiframe_metas):
        return super(LidarDetRubyVel, self).postprocess(multiframe_points)
