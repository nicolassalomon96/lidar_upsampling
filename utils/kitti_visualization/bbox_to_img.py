import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import imageio.v2 as io
from matplotlib.patches import Rectangle

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

  # load image
  img = np.array(io.imread(rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\image_2\{file_id}.png'), dtype=np.int32)
  # load labels
  with open(rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\label_2\{file_id}.txt', 'r') as f:
    labels = f.readlines()

  # load calibration file
  with open(rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\calib\{file_id}.txt', 'r') as f:
    lines = f.readlines()
    P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

  fig = plt.figure()
  # draw image
  plt.imshow(img)

  for line in labels:
    line = line.split()
    lab, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = line
    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
    if lab in names:
      plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2,
                                    edgecolor=colors[names.index(lab)],
                                    facecolor='none'))
      plt.text(x1 + 3, y1 + 3, lab,
               bbox=dict(facecolor=colors[names.index(lab)], alpha=0.5),
               fontsize=7, color='k')

  plt.axis('off')
  plt.tight_layout()
  #plt.savefig('examples/kitti_bbox_to_img.png', bbox_inches='tight')
  plt.show()
