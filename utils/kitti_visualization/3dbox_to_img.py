import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import imageio.v2 as io
from matplotlib.lines import Line2D

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
file_id = '000145'

if __name__ == '__main__':

  # load labels
  label_dir = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\PointPillars\dataset\kitti_64x2048_originales_no_ego_filtered\training\label_2\{file_id}.txt'
  with open(label_dir, 'r') as f:
    labels = f.readlines()

  # load image
  img = np.array(io.imread(rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\PointPillars\dataset\kitti_64x2048_originales_no_ego_filtered\training\image_2\{file_id}.png'), dtype=np.int32)

  # load calibration file
  with open(rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\PointPillars\dataset\kitti_64x2048_originales_no_ego_filtered\training\calib\{file_id}.txt', 'r') as f:
    lines = f.readlines()
    P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)


  # load image
  #img = np.array(io.imread(rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\image_2\{file_id}.png'), dtype=np.int32)

  # load labels
  #with open(rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\label_2\{file_id}.txt', 'r') as f:
  #  labels = f.readlines()

  # load calibration file
  #with open(rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\calib\{file_id}.txt', 'r') as f:
  #  lines = f.readlines()
  #  P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

  fig = plt.figure()
  # draw image
  plt.imshow(img)

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

      # transform the 3d bbox from camera_0 coordinate to camera_x image
      corners_3d_hom = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
      corners_img = np.matmul(corners_3d_hom, P2.T)
      corners_img = corners_img[:, :2] / corners_img[:, 2][:, None]


      def line(p1, p2, front=1):
        plt.gca().add_line(Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=colors[names.index(lab) * 2 + front]))


      # draw the upper 4 horizontal lines
      line(corners_img[0], corners_img[1], 0)  # front = 0 for the front lines
      line(corners_img[1], corners_img[2])
      line(corners_img[2], corners_img[3])
      line(corners_img[3], corners_img[0])

      # draw the lower 4 horizontal lines
      line(corners_img[4], corners_img[5], 0)
      line(corners_img[5], corners_img[6])
      line(corners_img[6], corners_img[7])
      line(corners_img[7], corners_img[4])

      # draw the 4 vertical lines
      line(corners_img[4], corners_img[0], 0)
      line(corners_img[5], corners_img[1], 0)
      line(corners_img[6], corners_img[2])
      line(corners_img[7], corners_img[3])

  # fig.patch.set_visible(False)
  plt.axis('off')
  plt.tight_layout()
  #plt.savefig('examples/kitti_3dbox_to_img.png', bbox_inches='tight')
  plt.show()
