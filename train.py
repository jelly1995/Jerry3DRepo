import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

cfg_file = "./configs/pointpillars_nuscenes.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/home/jerry/Desktop/NuScenesDataset/mini"
dataset = ml3d.datasets.NuScenes(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)


# get the weights.
ckpt_folder = "./logs/PointPillars_NuScenes_torch/checkpoint"
ckpt_path = os.path.join(ckpt_folder, "ckpt_00150.pth")
if not os.path.exists(ckpt_path):
    print('File not found')

pipeline.cfg_tb = {
    'readme': 'readme',
    'cmd_line': 'cmd_line',
    'dataset': '',
    'model': '',
    'pipeline': ''
}

pipeline.load_ckpt(ckpt_path=ckpt_path)


test_split = dataset.get_split("test")
data = test_split.get_data(0)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)
print(result)
pipeline.run_test()

#pipeline.run_train() 