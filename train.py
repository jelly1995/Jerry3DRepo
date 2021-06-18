import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from open3dml.ml3d.datasets.nuscenes import NuScenes as NS
from open3dml.ml3d.torch.pipelines.object_detection import ObjectDetection as OD
from open3dml.ml3d.torch.models.point_pillars import PointPillars as PP

cfg_file = "./open3dml/ml3d/configs/pointpillars_nuscenes.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
cfg.dataset['dataset_path'] = "/media/jerry/HDD/N"

model = PP(**cfg.model)
dataset = NS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = OD(model, dataset=dataset, device="gpu", **cfg.pipeline)

# get the weights.
""" ckpt_folder = "/home/jerry/Desktop/checkpoint"
ckpt_path = os.path.join(ckpt_folder, "ckpt_00115.pth")
if not os.path.exists(ckpt_path):
    print('File not found')  """
#pipeline.load_ckpt(ckpt_path=ckpt_path) 

pipeline.cfg_tb = {
    'readme': 'readme',
    'cmd_line': 'cmd_line',
    'dataset': '',
    'model': '',
    'pipeline': ''
}

pipeline.run_train()