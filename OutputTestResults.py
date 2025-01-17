import os
import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

import json
from pyquaternion import Quaternion

from open3dml.ml3d.datasets.nuscenes import NuScenes as NS
from open3dml.ml3d.torch.pipelines.object_detection import ObjectDetection as OD
from open3dml.ml3d.torch.models.point_pillars import PointPillars as PP


def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

cfg_file = "./open3dml/ml3d/configs/pointpillars_nuscenes.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
cfg.dataset['dataset_path'] = "/media/jerry/HDD/NTest"

model = PP(**cfg.model)
dataset = NS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = OD(model, dataset=dataset, device="gpu", **cfg.pipeline)

# get the weights.
ckpt_folder = "/home/jerry/Desktop/checkpoint"
ckpt_path = os.path.join(ckpt_folder, "ckpt_00100.pth")
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

data_split = dataset.get_split('test')

with open('results.json', 'w') as outfile:

    submission = {
        'meta' : {
            "use_camera" : False,
            "use_lidar" : True,
            "use_radar" : False,
            "use_map" : False,
            "use_external" : False,
        },
        'results' : {

        }
    }

    for i in range(6008):
        print(i)
        first_data = data_split.get_data_without_label(i)

        token = first_data['token']
        ego2global_tr = first_data['ego2global_tr']
        ego2global_rot = first_data['ego2global_rot']
        lidar2ego_tr = first_data['lidar2ego_tr']
        lidar2ego_rot = first_data['lidar2ego_rot']
        bounding_boxes = first_data['bounding_boxes']

        first_data = model.preprocess(first_data, {'split': 'test'})
        results = pipeline.run_inference(first_data)

        #print("Predicted boxes:" + str(len(results[0])))

        sample_results = []
        for bbox in results[0]:
            attr = ""
            if str(bbox.label_class) == "car":
                attr = "vehicle.parked"
            elif str(bbox.label_class) == "truck":
                attr = "vehicle.parked"
            elif str(bbox.label_class) == "pedestrian":
                attr = "pedestrian.standing"
            elif str(bbox.label_class) == "bicycle":
                attr = "cycle.with_rider"
            
            translation = bbox.center

            yaw = bbox.yaw
            rot = -(yaw + (np.pi / 2))
            QuadRot = euler_to_quaternion(rot, 0, 0)
            QuadRot = Quaternion(lidar2ego_rot) * QuadRot
            QuadRot = Quaternion(ego2global_rot) * QuadRot

            translation = np.dot(Quaternion(lidar2ego_rot).rotation_matrix, translation)
            translation += lidar2ego_tr
            translation = np.dot(Quaternion(ego2global_rot).rotation_matrix, translation)
            translation += ego2global_tr

            QuadRot[1] = 0
            QuadRot[2] = 0
            rotation = [r for r in QuadRot]

            temp = bbox.size[1]
            bbox.size[1] = bbox.size[2]
            bbox.size[2] = temp
  
            dictionary = {
                'sample_token' : str(token),
                'translation' : np.array(translation).tolist(),
                'size' : np.array(bbox.size).tolist(),
                'rotation' : np.array(rotation).tolist(),
                'velocity' : np.array([0, 0]).tolist(),
                'detection_name' : str(bbox.label_class),
                'detection_score' : bbox.confidence.tolist(),
                'attribute_name' : attr,
            }
            sample_results.append(dictionary)

        submission['results'][str(token)] = sample_results
        
    json.dump(submission, outfile, indent=4)


    """ json.dump(dictionary, outfile)
    outfile.write('\n') """
    
