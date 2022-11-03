import glob
import tqdm
import json
import os.path as osp
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from track_python import CTRACK
from deploy3d.symfun.models import LidarDetRuby, LidarDetOuster


def load_pose(pose_file):
    with open(pose_file, 'r') as f:
        pose_dict = json.load(f)
    
    _id = osp.splitext(osp.split(pose_file)[-1])[0]
    pose = pose_dict['poses']['/lidar/current_pose']
    timestamp = pose_dict[timestamp]
    
    return dict(_id=_id,
                pose=pose,
                timestamp=timestamp)

def main():
    track = CTRACK(isVel=False)
    
    lidar_det_ruby = LidarDetRuby()
    lidar_det_ouster = LidarDetOuster()

    json_path = ''
    pts_files = sorted(glob.glob('data/*.npy'))

    def vis_iter_fun():
        for i, pts_file in enumerate(tqdm.tqdm(pts_files)):
            pts = np.load(pts_file)
            result_ruby = lidar_det_ruby(pts)
            result_ouster = lidar_det_ouster(pts)
            labels_3d = result_ruby['labels_3d'] + result_ouster['labels_3d']
            scores_3d = np.concatenate([result_ruby['scores_3d'], 
                                        result_ouster['scores_3d']], axis=0)
            boxes_3d = np.concatenate([result_ruby['boxes_3d'], 
                                        result_ouster['boxes_3d']], axis=0)
            res_info = dict(labels_3d=labels_3d,
                            scores_3d=scores_3d,
                            boxes_3d=boxes_3d)
            
            pose_file = osp.join(json_path, osp.split(pts_file)[-1] + '.json')
            pose_info = load_pose(pose_file)
            
            track.TrackProc(res_info, pose_info)
            track_data = track.GetTrackData()
            track_data['labels_3d'] = np.array([lidar_det_ruby.classes.index(l) 
                                                for l in track_data['labels_3d']])
            
            yield i, pts, track_data

    vis_iter = vis_iter_fun()

    def box3d_corners(boxes):
        if boxes.shape[0] == 0:
            return np.empty([0, 8, 3], dtype=boxes.dtype)

        dims = boxes[:, 3:6]
        corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - np.array([0.5, 0.5, 0], dtype=dims.dtype)
        corners = dims.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        angles = boxes[:, 6]
        rot_sin = np.sin(angles)
        rot_cos = np.cos(angles)
        ones = np.ones_like(rot_cos)
        zeros = np.zeros_like(rot_cos)
        rot_mat_T = np.stack([
            np.stack([rot_cos, rot_sin, zeros]),
            np.stack([-rot_sin, rot_cos, zeros]),
            np.stack([zeros, zeros, ones])
        ])
        corners = np.einsum('aij,jka->aik', corners, rot_mat_T)
        corners += boxes[:, :3].reshape(-1, 1, 3)
        return corners

    def renderbox(box3d, labels, color=None):
        clr_map = plt.get_cmap('tab10').colors
        corners = box3d_corners(box3d)
        cores = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (8, 4), (8, 5), (8, 6), (8, 7)
        ]
        ret = None
        for corners_i, label_i in zip(corners, labels):
            corners_i = corners_i.astype(np.float64)
            frontcenter = corners_i[[4, 5, 6, 7]].mean(axis=0, keepdims=True)
            heading = corners_i[4] - corners_i[0]
            frontcenter += 0.3 * heading / np.linalg.norm(heading)
            corners_i = np.concatenate((corners_i, frontcenter), axis=0)
            corners_i = o3d.utility.Vector3dVector(corners_i)
            corners_i = o3d.geometry.PointCloud(points=corners_i)

            box = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                corners_i,
                corners_i,
                cores)
            if color is None:
                box.paint_uniform_color(clr_map[label_i % len(clr_map)])
            else:
                box.paint_uniform_color(color)
            if ret is None:
                ret = box
            else:
                ret += box

        return ret

    def key_cbk(vis: o3d.visualization.Visualizer):
        try:
            idx, points, track_data = next(vis_iter)
        except StopIteration:
            return True

        det_box3d = np.stack(track_data['boxes_3d'], axis=0)[:, 0:7]
        det_names = track_data['labels_3d'].astype(np.uint8)
        det_scores = np.array(track_data['scores_3d'])
        track_id = np.array(track_data['track_id'], dtype=np.int32)

        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1).astype(np.float64)
        clr = plt.get_cmap('gist_rainbow')(points['intensity'])[:, :3]

        points = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(xyz))
        points.colors = o3d.utility.Vector3dVector(clr)

        vis.clear_geometries()
        vis.add_geometry(points, idx == 0)

        if det_box3d is not None and len(det_box3d):
            det_box = renderbox(det_box3d, det_names)
            vis.add_geometry(det_box, idx == 0)

        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(" "), key_cbk)
    vis.create_window(width=1080, height=720)
    op = vis.get_render_option()
    op.background_color = np.array([0., 0., 0.])
    op.point_size = 2.0
    if key_cbk(vis):
        return
    else:
        vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
