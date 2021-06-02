import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

dataset = ml3d.datasets.NuScenes(dataset_path='/home/jerry/Desktop/NuScenesDataset/mini')

all_split = dataset.get_split('train')

print(all_split)

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)

# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, 'train', indices=range(10))