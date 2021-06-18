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

def ToQuaternion(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w,x,y,z]


cfg_file = "./open3dml/ml3d/configs/pointpillars_nuscenes.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
cfg.dataset['dataset_path'] = "/media/jerry/HDD/NMini"

model = PP(**cfg.model)
dataset = NS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = OD(model, dataset=dataset, device="gpu", **cfg.pipeline)

data_split = dataset.get_split('val')

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

    for i in range(3):
        print(i)
        first_data = data_split.get_data(i)

        token = first_data['token']
        ego2global_tr = first_data['ego2global_tr']
        ego2global_rot = first_data['ego2global_rot']
        lidar2ego_tr = first_data['lidar2ego_tr']
        lidar2ego_rot = first_data['lidar2ego_rot']
        bounding_boxes = first_data['bounding_boxes']

        first_data = model.preprocess(first_data, {'split': 'test'})
        results = pipeline.run_inference(first_data)

        sample_results = []

        

        for bbox in bounding_boxes:    
            translation = bbox.center

            if token == '3e8750f331d7499e9b5123e9eb70f2e2':
                
                rotation_test = Quaternion([0.3598258294147673, 0.0, 0.0, 0.9330194920182401])
                print("Ground Truth Starting:" + str(rotation_test))
                rotation_test = Quaternion(ego2global_rot).inverse * rotation_test
                rotation_test = Quaternion(lidar2ego_rot).inverse * rotation_test

                rot_test = rotation_test.yaw_pitch_roll[0]
                rot_test = - rot_test - np.pi / 2
            
                print("Calculated Ending:" + str(rot_test)) 

                yaw = bbox.yaw
                print("Ground Truth Ending:" + str(yaw))

                rot = -(yaw + (np.pi / 2))
                rotation = ToQuaternion(rot, 0, 0)

                rotation = Quaternion(lidar2ego_rot) * rotation
                rotation = Quaternion(ego2global_rot) * rotation
                print(rotation)




                #print("-----------------------------------------------------------")

            QuadRot = euler_to_quaternion(yaw, 0, 0)

            translation = np.dot(Quaternion(lidar2ego_rot).rotation_matrix, translation)
            QuadRot = Quaternion(lidar2ego_rot) * QuadRot

            translation += lidar2ego_tr

            translation = np.dot(Quaternion(ego2global_rot).rotation_matrix, translation)
            QuadRot = Quaternion(ego2global_rot) * QuadRot

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
            }
            sample_results.append(dictionary)

        submission['results'][str(token)] = sample_results
        
    json.dump(submission, outfile, indent=4)


    """ json.dump(dictionary, outfile)
    outfile.write('\n') """
    
