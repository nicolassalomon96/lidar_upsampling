import numpy as np
import seaborn as sns
import mayavi.mlab as mlab
import sys

#colors = sns.color_palette('Paired', 9 * 2)
#names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
colors = sns.color_palette('Paired', 3*2)
colors = colors[::-1]
names = list(LABEL2CLASSES.values())

file_id = '001648'

if __name__ == '__main__':

  # load point clouds
  scan_dir = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\PointPillars\dataset\kitti_64x2048_originales_no_ego_filtered\training\velodyne\{file_id}.bin'
  scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)

  # load labels
  label_dir = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\PointPillars\dataset\kitti_64x2048_originales_no_ego_filtered\training\label_2\{file_id}.txt'
  with open(label_dir, 'r') as f:
    labels = f.readlines()

  fig = mlab.figure(bgcolor=(0.27, 0.27, 0.54), size=(1280, 720))
  # draw point cloud
  plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig, color=(1,1,0))

  for line in labels:
    line = line.split()
    #lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, _ = line
    lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line

    h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
    if lab in names:
      x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
      y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
      z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
      corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

      # transform the 3d bbox from object coordiante to camera_0 coordinate
      R = np.array([[np.cos(rot), 0, np.sin(rot)],
                    [0, 1, 0],
                    [-np.sin(rot), 0, np.cos(rot)]])
      corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

      # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
      corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])


      def draw(p1, p2, front=1):
        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=colors[names.index(lab) * 2 + front], tube_radius=None, line_width=2, figure=fig)


      # draw the upper 4 horizontal lines
      draw(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
      draw(corners_3d[1], corners_3d[2])
      draw(corners_3d[2], corners_3d[3])
      draw(corners_3d[3], corners_3d[0])

      # draw the lower 4 horizontal lines
      draw(corners_3d[4], corners_3d[5], 0)
      draw(corners_3d[5], corners_3d[6])
      draw(corners_3d[6], corners_3d[7])
      draw(corners_3d[7], corners_3d[4])

      # draw the 4 vertical lines
      draw(corners_3d[4], corners_3d[0], 0)
      draw(corners_3d[5], corners_3d[1], 0)
      draw(corners_3d[6], corners_3d[2])
      draw(corners_3d[7], corners_3d[3])

  mlab.view(azimuth=230, distance=50)
  #mlab.savefig(filename='examples/kitti_3dbox_to_cloud.png')
  mlab.show()
