import glob
from deploy3d.symfun.models import LidarSegRubyOuster
import tqdm
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def dbscan(points, labels, min_points):
    # points that will be clustered
    cluster_labels = [4, 6, 7, 8, 9, 10, 13, 14]
    
    # eps value for each label points
    eps_list = [1, 1, 1, 1, 1, 1, 0.5, 1]
    min_points = 10
    
    # num iters for each label
    iter_nums = [1, 1, 1, 1, 1, 1, 1, 1]
    
    xyz = np.stack([points['x'], points['y'], points['z']], axis=1)
    cluster_ids = np.zeros(points.shape[0], dtype=np.int16)
    
    offset_value = 1 # cluster id start from 1, due to only cluster part of labels
    for cluster_label, eps, iters in zip(cluster_labels, eps_list, iter_nums):
        mask = (labels==cluster_label)
        if not mask.any():
            continue
        
        pts = xyz[mask]
        pts_labels = labels[mask]
        label_ids = [cluster_label]
        
        pts_clusters = np.zeros(pts.shape[0], dtype=np.int16)
        for idx in range(iters):
            offset = 0
            for label_id in label_ids:
                label_mask = (pts_labels==label_id)
                if not label_mask.any():
                    continue
                
                points = pts[label_mask]
                pcd = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(points))
                
                cluster_id = np.array(pcd.cluster_dbscan(eps/(idx+1), min_points), dtype=np.int16)
                num_clusters = cluster_id.max() + 1
                
                cluster_id = np.where(cluster_id==-1, -1, cluster_id + offset)
                
                pts_clusters[label_mask] = cluster_id
                offset += num_clusters
            
            label_ids = np.arange(pts_clusters.max() + 1, dtype=np.int16)
            pts_labels = pts_clusters
        
        num_this_label_clusters = pts_clusters.max() + 1
        pts_clusters = np.where(pts_clusters==-1, -1, pts_clusters + offset_value)
        cluster_ids[mask] = pts_clusters
        offset_value += num_this_label_clusters
    return cluster_ids

def main():
    lidar_cylinder3d = LidarSegRubyOuster()
    
    pts_files = sorted(glob.glob('data/*.npy'))
    
    def vis_iter_fun():
        for i, pts_file in enumerate(tqdm.tqdm(pts_files)):
            raw_points = np.load(pts_file)
            result_ruby = lidar_cylinder3d(raw_points)
            # seg_mask is valid input point_idx
            seg_labels, seg_mask = result_ruby['seg_labels'], result_ruby['seg_mask']
            
            num_pts = min(seg_mask.shape[0], 480000)
            seg_mask = seg_mask[0:num_pts]
            
            points = raw_points[seg_mask]
            labels = seg_labels[0:num_pts]
            
            single_cluster_id = 0 # single label for clustering
            if not single_cluster_id: # clustering all labels, or don't cluster
                cluster_ids = None
                if False: # whether do dbscan
                    cluster_ids = dbscan(points, labels, min_points=10)
                yield i, points, labels, cluster_ids
            
            else:
                mask = (labels==single_cluster_id)
                points, labels = points[mask], labels[mask]
                cluster_ids = dbscan(points, labels, min_points=10)
                yield i, points, labels, cluster_ids
            

    vis_iter = vis_iter_fun()

    def key_cbk(vis: o3d.visualization.Visualizer):
        try:
            idx, points, labels, cluster_ids = next(vis_iter)
        except StopIteration:
            return True
        
        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1).astype(np.float64)
        if cluster_ids is not None:
            max_id = cluster_ids.max()
            labels =  np.where(cluster_ids==-1, max_id+1, cluster_ids)
            max_value = labels.max()
            
            if (max_value == max_id): # has no noise points
                max_value += 1
            # noise pts is white
            clr = plt.get_cmap('gist_ncar')(labels / max_value)[:, :3]
        
        else:
            labels =  labels.astype(np.uint8)
            clr = plt.get_cmap('tab20')(labels)[:, :3]
        
        points = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(xyz))
        points.colors = o3d.utility.Vector3dVector(clr)

        vis.clear_geometries()
        vis.add_geometry(points, idx == 0)

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
