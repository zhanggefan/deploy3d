import glob
import tqdm
import time
import json
import numpy as np
import open3d as o3d
import os.path as osp
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui

from track_python import CTRACK
from deploy3d.symfun.models import LidarDetRubyVel
from functools import partial


def vis_iter_fun(pts_files, pts_metas, lidar_det_ruby_vel, track):
    for i, pts_file in enumerate(tqdm.tqdm(pts_files)):
        pts = [np.load(_pts) for _pts in pts_file]
        metas = [pts_metas[osp.split(_pts)[-1]] for _pts in pts_file]
        result_ruby = lidar_det_ruby_vel(pts, metas)
        
        labels_3d = result_ruby['labels_3d']
        boxes_3d = result_ruby['boxes_3d']
        labels_3d = np.array([lidar_det_ruby_vel.classes.index(l) 
                                for l in labels_3d])
        
        pose_info = dict(_id=metas[0]['_id'],
                         pose=metas[0]['pose'],
                         timestamp=eval(metas[0]['timestamp']['$numberLong']))
        
        track.TrackProc(result_ruby, pose_info)
        track_data, _ = track.GetTrackData()
        track_data['labels_3d'] = np.array([lidar_det_ruby_vel.classes.index(l) 
                                            for l in track_data['labels_3d']])
        
        yield i, pts[0], track_data, (boxes_3d, labels_3d)

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

def rendertrackbox(box3d, labels, track_ids, color=(0,1,0)):
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
    text_info = []
    for i, (corners_i, label_i, track_id) in enumerate(zip(corners, labels, track_ids)):
        corners_i = corners_i.astype(np.float64)
        frontcenter = corners_i[[4, 5, 6, 7]].mean(axis=0, keepdims=True)
        heading = corners_i[4] - corners_i[0]
        frontcenter += 0.3 * heading / np.linalg.norm(heading)
        corners_i = np.concatenate((corners_i, frontcenter), axis=0)
        text_info.append((corners_i[0:8,:].mean(axis=0), f'{track_id}'))
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
                if color is None:
                    vel_vector.paint_uniform_color(
                        clr_map[label_i % len(clr_map)])
                else:
                    vel_vector.paint_uniform_color(color)
                
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

    return ret, vel_vectors, text_info

def renderdetbox(box3d, labels, color=(0,0,1)):
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
                if color is None:
                    vel_vector.paint_uniform_color(
                        clr_map[label_i % len(clr_map)])
                else:
                    vel_vector.paint_uniform_color(color)
                
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

def update_thread(vis_iter, vis):
    is_done = False
    while not is_done:
        time.sleep(1)
        
        try:
            _, points, track_data, res_info = next(vis_iter)
        except:
            is_done = True
        
        track_box3d = np.stack(track_data['boxes_3d'], axis=0)
        track_names = track_data['labels_3d'].astype(np.uint8)
        track_scores = np.array(track_data['scores_3d'])
        track_ids = np.array(track_data['track_id'], dtype=np.int32)

        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1).astype(np.float64)
        clr = plt.get_cmap('gist_rainbow')(points['intensity'])[:, :3]

        def render():
            points = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(xyz))
            points.colors = o3d.utility.Vector3dVector(clr)

            vis.remove_geometry('points')
            vis.add_geometry('points', points)

            if track_box3d is not None and len(track_box3d):
                track_box, track_vel, text_info = rendertrackbox(track_box3d, track_names, track_ids, color=(0,1,0))
                vis.remove_geometry('track_box')
                vis.add_geometry('track_box', track_box)
                
                if track_vel is not None:
                    vis.remove_geometry('track_vel')
                    vis.add_geometry('track_vel', track_vel)
                    
                vis.clear_3d_labels()
                for info in text_info:
                    pose, text = info
                    vis.add_3d_label(pose, text)

            if res_info is not None and len(res_info):
                boxes_3d, labels_3d = res_info
                det_box, det_vel = renderdetbox(boxes_3d, labels_3d, color=(0,0,1))
                
                vis.remove_geometry('det_box')
                vis.add_geometry('det_box', det_box)
                
                if det_vel is not None:
                    vis.remove_geometry('det_vel')
                    vis.add_geometry('det_vel', det_vel)
            
            vis.post_redraw()
        
        if not is_done:
            gui.Application.instance.post_to_main_thread(vis, render)

def main():
    pts_files = sorted(glob.glob('data/*.npy'))
    pts_metas = './data/metas.json'
    
    isVel = True # whether with vel for tracking
    track = CTRACK(isVel=isVel)
    
    lidar_det_ruby_vel = LidarDetRubyVel()

    with open(pts_metas) as f:
        pts_metas = {m['_id']: m for m in json.load(f)}
    pts_files = [*zip(pts_files[1:], pts_files[:-1])]

    vis_iter = vis_iter_fun(pts_files, pts_metas, lidar_det_ruby_vel, track)

    app = gui.Application.instance
    app.initialize()
    
    vis = o3d.visualization.O3DVisualizer()
    vis.enable_raw_mode(True)
    vis.set_background((255.0, 255.0, 255.0, 255.0), None)
    vis.point_size = 2
    vis.setup_camera(60, (0, 0, 0), (-20, 0, 20), (1, 0, 1))
    vis.show_skybox(False)
    vis.show_ground = False
    vis.ground_plane = vis.ground_plane.XY
    
    app.add_window(vis)
    app.run_in_thread(partial(update_thread, vis_iter, vis))
    app.run()

if __name__ == '__main__':
    main()
