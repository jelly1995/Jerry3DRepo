import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

""" # construct a dataset by specifying dataset_path
dataset = ml3d.datasets.NuScenes(dataset_path='/home/jerry/Desktop/NuScenesDataset/mini')

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('val')
print(all_split)
# print the attributes of the first datum
#print(all_split.get_attr(0))

# print the shape of the first point cloud
#print(all_split.get_data(0)['point'].shape)

# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, 'val', indices=range(10))
 """



cfg_file = "./Open3D-ML/ml3d/configs/pointpillars_nuscenes.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/home/jerry/Desktop/NuScenesDataset/testing"
dataset = ml3d.datasets.NuScenes(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# get the weights.
ckpt_folder = "./logs/PointPillars_NuScenes_torch/checkpoint"
ckpt_path = os.path.join(ckpt_folder, "ckpt_00150.pth")
if not os.path.exists(ckpt_path):
    print("Not found")

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

""" training_split = dataset.get_split("train")

data = training_split.get_data(0)
print((data['point'].shape))

data = training_split.get_data(1)
print((data['point'].shape))

test_split = dataset.get_split("test")

data = test_split.get_data(0)
print((data['point'].shape)) """

""" test_split = dataset.get_split("test")
data = test_split.get_data(0)
print(data) """

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
#result = pipeline.run_inference(data)

#print(result)

# evaluate performance on the test set; this will write logs to './logs'.
pipeline.run_test() 