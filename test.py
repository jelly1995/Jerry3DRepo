import os
import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

#import open3d.ml.torch as ml3d
from ml3d.ml3d.datasets.nuscenes import NuScenes as NS
from ml3d.ml3d.torch.pipelines.object_detection import ObjectDetection as OD
from ml3d.ml3d.torch.models.point_pillars import PointPillars as PP

from ml3d.ml3d.vis.labellut import *

cfg_file = "./ml3d/ml3d/configs/pointpillars_nuscenes.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = PP(**cfg.model)
cfg.dataset['dataset_path'] = "/home/jerry/Desktop/NuScenesDataset/mini"
dataset = NS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = OD(model, dataset=dataset, device="gpu", **cfg.pipeline)

# get the weights.
ckpt_folder = "./logs/PointPillars_NuScenes_torch/checkpoint"
ckpt_path = os.path.join(ckpt_folder, "ckpt_00150.pth")
if not os.path.exists(ckpt_path):
    print("Not found")

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)
pipeline.cfg_tb = {
    'readme': 'readme',
    'cmd_line': 'cmd_line',
    'dataset': '',
    'model': '',
    'pipeline': ''
}

#boxes, pred, gt = pipeline.run_self_inference()

data_split = dataset.get_split('train')
first_data = data_split.get_data(5)

new_data = model.preprocess(first_data, {'split': 'test'})

results = pipeline.run_inference(new_data)

print("Predicted boxes:" + str(len(results[0])))
#print(results[0])
predictedLabels = np.array([results[0][i].label_class for i in range(len(results[0]))])
#print(predictedLabels)
print("----------------------------------------------")
print("Ground truth boxes:" + str(len(first_data['bounding_boxes'])))

vis = ml3d.vis.Visualizer()
data = [ {
    'name': 'my_point_cloud',
    'points': first_data['point'],
} ]

#vis.visualize(data=data, bounding_boxes=results[0])
vis.visualize(data=data, bounding_boxes=results[0])