import glob
from deploy3d.symfun.models import LidarDetRubyVel
import tqdm
import numpy as np
import open3d as o3d
import os.path as osp
import matplotlib.pyplot as plt
import json


def main():
    pts_files = sorted(glob.glob('data/*.npy'))
    pts_metas = './data/metas.json'

    lidar_det_ruby_vel = LidarDetRubyVel()

    with open(pts_metas) as f:
        pts_metas = {m['_id']: m for m in json.load(f)}
    pts_files = [*zip(pts_files[1:], pts_files[:-1])]

    def vis_iter_fun():
        for i, pts_file in enumerate(tqdm.tqdm(pts_files)):
            pts = [np.load(_pts) for _pts in pts_file]
            metas = [pts_metas[osp.split(_pts)[-1]] for _pts in pts_file]
            result_ruby = lidar_det_ruby_vel(pts, metas)
            result_ruby['labels_3d'] = np.array(
                [lidar_det_ruby_vel.classes.index(l) for l in result_ruby['labels_3d']])
            yield i, pts[0], [result_ruby]

    vis_iter = vis_iter_fun()

    def box3d_corners(boxes):
        if boxes.shape[0] == 0:
            return np.empty([0, 8, 3], dtype=boxes.dtype)

        dims = boxes[:, 3:6]
        corners_norm = np.stack(np.unravel_index(
            np.arange(8), [2] * 3), axis=1)

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
        if box3d.shape[-1] == 9:
            vels = box3d[:, -2:]
            vels_norm = np.linalg.norm(vels, axis=-1, ord=2)
            vels_yaw = np.arctan2(vels[:, 1], vels[:, 0])
            vels_ctr = box3d[:, :3]
        cores = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (8, 4), (8, 5), (8, 6), (8, 7)
        ]
        ret = None
        vel_vectors = None
        for i, (corners_i, label_i) in enumerate(zip(corners, labels)):
            corners_i = corners_i.astype(np.float64)
            frontcenter = corners_i[[4, 5, 6, 7]].mean(axis=0, keepdims=True)
            heading = corners_i[4] - corners_i[0]
            frontcenter += 0.3 * heading / np.linalg.norm(heading)
            corners_i = np.concatenate((corners_i, frontcenter), axis=0)
            corners_i = o3d.utility.Vector3dVector(corners_i)
            corners_i = o3d.geometry.PointCloud(points=corners_i)

            if box3d.shape[-1] == 9:  # with velocity
                vel_norm = vels_norm[i]
                vel_yaw = vels_yaw[i]
                if vel_norm > 0:
                    vel_vector = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=0.1, cone_radius=0.3,
                        cylinder_height=vel_norm, cone_height=0.5)
                    R = vel_vector.get_rotation_matrix_from_xyz(
                        (0, np.pi / 2, 0))
                    vel_vector.rotate(R, center=(0, 0, 0))
                    R = vel_vector.get_rotation_matrix_from_xyz(
                        (0, 0, vel_yaw))
                    vel_vector.rotate(R, center=(0, 0, 0))
                    vel_vector.translate(vels_ctr[i])
                    vel_vector.paint_uniform_color(
                        clr_map[label_i % len(clr_map)])
                    if vel_vectors is None:
                        vel_vectors = vel_vector
                    else:
                        vel_vectors += vel_vector

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

        return ret, vel_vectors

    def key_cbk(vis: o3d.visualization.Visualizer):
        try:
            idx, points, det = next(vis_iter)
        except StopIteration:
            return True

        det_box3d = np.concatenate([d['boxes_3d'] for d in det], axis=0)
        det_names = np.concatenate([d['labels_3d'] for d in det], axis=0)
        det_scores = np.concatenate([d['scores_3d'] for d in det], axis=0)

        xyz = np.stack([points['x'], points['y'], points['z']],
                       axis=-1).astype(np.float64)
        clr = plt.get_cmap('gist_rainbow')(points['intensity'])[:, :3]

        points = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(xyz))
        points.colors = o3d.utility.Vector3dVector(clr)

        vis.clear_geometries()
        vis.add_geometry(points, idx == 0)

        if det_box3d is not None and len(det_box3d):
            det_box, det_vel = renderbox(det_box3d, det_names)
            vis.add_geometry(det_box, idx == 0)
            if det_vel is not None:
                vis.add_geometry(det_vel, idx == 0)

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
