import glob
from deploy3d.symfun.models import LidarSegRubyOuster
import tqdm
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def main():
    cylinder3d_path = 'seg3d_ruby_ouster'

    lidar_cylinder3d = LidarSegRubyOuster(cylinder3d_path)
    
    pts_files = sorted(glob.glob('data/*.npy'))
    
    def vis_iter_fun():
        for i, pts_file in enumerate(tqdm.tqdm(pts_files)):
            pts = np.load(pts_file)
            result_ruby = lidar_cylinder3d(pts)
            labels, mask = result_ruby['labels'], result_ruby['mask']
            yield i, pts[mask], labels

    vis_iter = vis_iter_fun()

    def key_cbk(vis: o3d.visualization.Visualizer):
        try:
            idx, points, labels = next(vis_iter)
        except StopIteration:
            return True
        
        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1).astype(np.float64)
        labels = labels[:points.shape[0]].astype(np.uint8)
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
