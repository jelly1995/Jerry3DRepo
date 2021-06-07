import os
from ml3d.ml3d.datasets.nuscenes import NuScenes as NS
import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
import numpy as np

from ml3d.ml3d.vis.visualizer import Visualizer as V


dataset = NS(dataset_path='/home/jerry/Desktop/NuScenesDataset/mini')

train_split = dataset.get_split('train')
first_data = train_split.get_data(0)
boxes = first_data['bounding_boxes']


# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()

data = [ {
    'name': 'my_point_cloud',
    'points': first_data['point'],
} ]

#vis.visualize_dataset(dataset, 'train', indices=range(10))

vis.visualize(data=data, bounding_boxes=boxes)
