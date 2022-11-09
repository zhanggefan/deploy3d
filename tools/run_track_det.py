import glob
import tqdm
import json
import time
import os.path as osp
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import matplotlib.pyplot as plt

from track_python import CTRACK
from deploy3d.symfun.models import LidarDetRuby, LidarDetOuster
from functools import partial

def load_pose(pose_file):
    with open(pose_file, 'r') as f:
        pose_dict = json.load(f)
    
    _id = osp.splitext(osp.split(pose_file)[-1])[0]
    pose = pose_dict['poses']['/lidar/current_pose']
    timestamp = pose_dict['timestamp']
    
    return dict(_id=_id,
                pose=pose,
                timestamp=timestamp)

def vis_iter_fun(pts_files, json_path, track, lidar_det_ruby, lidar_det_ouster):
    for i, pts_file in enumerate(tqdm.tqdm(pts_files)):
        labels_3d, scores_3d, boxes_3d = [], [], []
        pts = np.load(pts_file)
        if lidar_det_ruby is not None:
            result_ruby = lidar_det_ruby(pts)
            labels_3d.extend(result_ruby['labels_3d'])
            scores_3d.append(result_ruby['scores_3d'])
            boxes_3d.append(result_ruby['boxes_3d'])
        
        if lidar_det_ouster is not None:
            result_ouster = lidar_det_ouster(pts)
            labels_3d.extend(result_ouster['labels_3d'])
            scores_3d.append(result_ouster['scores_3d'])
            boxes_3d.append(result_ouster['boxes_3d'])
        
        scores_3d = np.concatenate(scores_3d, axis=0)
        boxes_3d = np.concatenate(boxes_3d, axis=0)
        res_info = dict(labels_3d=labels_3d,
                        scores_3d=scores_3d,
                        boxes_3d=boxes_3d)
        labels_3d = np.array([lidar_det_ruby.classes.index(l) 
                                for l in labels_3d])
        
        pose_file = osp.join(json_path, osp.split(pts_file)[-1] + '.json')
        pose_info = load_pose(pose_file)
        
        track.TrackProc(res_info, pose_info)
        track_data, _ = track.GetTrackData()
        track_data['labels_3d'] = np.array([lidar_det_ruby.classes.index(l) 
                                            for l in track_data['labels_3d']])
                
        yield i, pts, track_data, (boxes_3d, labels_3d)

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

def rendertrackbox(box3d, labels, track_ids, color=None):
    clr_map = plt.get_cmap('tab10').colors
    corners = box3d_corners(box3d)
    cores = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (8, 4), (8, 5), (8, 6), (8, 7)
    ]
    ret = None
    text_info = []
    for corners_i, label_i, track_id in zip(corners, labels, track_ids):
        corners_i = corners_i.astype(np.float64)
        frontcenter = corners_i[[4, 5, 6, 7]].mean(axis=0, keepdims=True)
        heading = corners_i[4] - corners_i[0]
        frontcenter += 0.3 * heading / np.linalg.norm(heading)
        corners_i = np.concatenate((corners_i, frontcenter), axis=0)
        text_info.append((corners_i[0:8,:].mean(axis=0), f'{track_id}'))
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

    return ret, text_info

def renderdetbox(box3d, labels, color=(0, 0, 1)):
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
                track_box, text_info = rendertrackbox(track_box3d, track_names, track_ids, color=(0,1,0))
                vis.remove_geometry('track_box')
                vis.add_geometry('track_box', track_box)
                
                vis.clear_3d_labels()
                for info in text_info:
                    pose, text = info
                    vis.add_3d_label(pose, text)
            
            if res_info is not None and len(res_info):
                boxes_3d, labels_3d = res_info
                det_box = renderdetbox(boxes_3d, labels_3d, color=(0,0,1))
                vis.remove_geometry('det_box')
                vis.add_geometry('det_box', det_box)
            
            vis.post_redraw()
        
        if not is_done:
            gui.Application.instance.post_to_main_thread(vis, render)

def main():
    json_path = '/data/output1/tag'
    data_path = '/data/output1/data'
    pts_files = sorted(glob.glob(osp.join(data_path, '*.npy')))
    
    isVel = False # whether with vel for tracking
    track = CTRACK(isVel=isVel)
    lidar_det_ruby = LidarDetRuby()
    lidar_det_ouster = LidarDetOuster()

    vis_iter = vis_iter_fun(pts_files, json_path, track, lidar_det_ruby, lidar_det_ouster=None)

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
