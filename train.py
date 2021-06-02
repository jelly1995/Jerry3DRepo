import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d



"""
# construct a dataset by specifying dataset_path
dataset = ml3d.datasets.NuScenes(dataset_path='/home/jerry/Desktop/NuScenesDataset/mini')

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('training')

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)

# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, 'training', indices=range(100)) """



cfg_file = "./Open3D-ML/ml3d/configs/pointpillars_nuscenes.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

# get the weights.
ckpt_folder = "./logs/PointPillars_NuScenes_torch/checkpoint"
ckpt_path = os.path.join(ckpt_folder, "ckpt_00140.pth")
if not os.path.exists(ckpt_path):
    print('File not found')
else:
    print('File found')


model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/home/jerry/Desktop/NuScenesDataset/mini"
dataset = ml3d.datasets.NuScenes(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)
pipeline.cfg_tb = {
    'readme': 'readme',
    'cmd_line': 'cmd_line',
    'dataset': '',
    'model': '',
    'pipeline': ''
}

pipeline.load_ckpt(ckpt_path=ckpt_path)

pipeline.run_train()