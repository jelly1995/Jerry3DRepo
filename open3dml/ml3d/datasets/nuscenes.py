import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml
from scipy.spatial.transform import Rotation as R

from .base_dataset import BaseDataset
from ..utils import Config, make_dir, DATASET
from .utils import BEVBox3D

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class NuScenes(BaseDataset):
    """
    NuScenes 3D dataset for Object Detection, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 info_path=None,
                 name='NuScenes',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
            kwargs:
        """

        if info_path is None:
            info_path = dataset_path

        super().__init__(dataset_path=dataset_path,
                         info_path=info_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 10
        self.label_to_names = self.get_label_to_names()

        self.train_info = {}
        self.test_info = {}
        self.val_info = {}

        if os.path.exists(join(info_path, 'infos_train.pkl')):
            self.train_info = pickle.load(
                open(join(info_path, 'infos_train.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_val.pkl')):
            self.val_info = pickle.load(
                open(join(info_path, 'infos_val.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_test.pkl')):
            self.test_info = pickle.load(
                open(join(info_path, 'infos_test.pkl'), 'rb'))

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'ignore',
            1: 'barrier',
            2: 'bicycle',
            3: 'bus',
            4: 'car',
            5: 'construction_vehicle',
            6: 'motorcycle',
            7: 'pedestrian',
            8: 'traffic_cone',
            9: 'trailer',
            10: 'truck',
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def read_label(info, calib):
        mask = info['num_lidar_pts'] != 0
        boxes = info['gt_boxes'][mask]
        names = info['gt_names'][mask]

        objects = []
        for name, box in zip(names, boxes):

            center = [float(box[0]), float(box[1]), float(box[2])]
            size = [float(box[3]), float(box[5]), float(box[4])]
            ry = float(box[6])

            yaw = ry - np.pi
            yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi

            world_cam = calib['world_cam']

            objects.append(BEVBox3D(center, size, yaw, name, -1.0, world_cam))
            objects[-1].yaw = ry

        return objects

    def get_split(self, split):
        return NuSceneSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['train', 'training']:
            return self.train_info
        elif split in ['test', 'testing']:
            return self.test_info
        elif split in ['val', 'validation']:
            return self.val_info

        raise ValueError("Invalid split {}".format(split))

    def is_tested():
        pass

    def save_test_result():
        pass


class NuSceneSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        self.infos = dataset.get_split_list(split)
        self.path_list = []
        self.token_list = []
        for info in self.infos:
            self.path_list.append(info['lidar_path'])
            self.token_list.append(info['token'])

        log.info("Found {} pointclouds for {}".format(len(self.infos), split))

        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.infos)

    def get_data(self, idx):
        info = self.infos[idx]
        lidar_path = info['lidar_path']
        token = info['token']

        world_cam = np.eye(4)
        world_cam[:3, :3] = R.from_quat(info['lidar2ego_rot']).as_matrix()
        world_cam[:3, -1] = info['lidar2ego_tr']
        calib = {'world_cam': world_cam.T}

        pc = self.dataset.read_lidar(lidar_path)
        label = self.dataset.read_label(info, calib)

        ego2global_tr = info['ego2global_tr']
        ego2global_rot = info['ego2global_rot']
        lidar2ego_tr = info['lidar2ego_tr']
        lidar2ego_rot = info['lidar2ego_rot']

        data = {
            'point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': label,
            'token': token,
            'ego2global_tr' : ego2global_tr,
            'ego2global_rot' : ego2global_rot,
            'lidar2ego_tr' : lidar2ego_tr,
            'lidar2ego_rot' : lidar2ego_rot,
        }

        return data

    def get_data_without_label(self, idx):
        info = self.infos[idx]
        lidar_path = info['lidar_path']
        token = info['token']

        world_cam = np.eye(4)
        world_cam[:3, :3] = R.from_quat(info['lidar2ego_rot']).as_matrix()
        world_cam[:3, -1] = info['lidar2ego_tr']
        calib = {'world_cam': world_cam.T}

        pc = self.dataset.read_lidar(lidar_path)
        
        ego2global_tr = info['ego2global_tr']
        ego2global_rot = info['ego2global_rot']
        lidar2ego_tr = info['lidar2ego_tr']
        lidar2ego_rot = info['lidar2ego_rot']

        data = {
            'point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': '',
            'token': token,
            'ego2global_tr' : ego2global_tr,
            'ego2global_rot' : ego2global_rot,
            'lidar2ego_tr' : lidar2ego_tr,
            'lidar2ego_rot' : lidar2ego_rot,
        }

        return data

    def get_attr(self, idx):
        info = self.infos[idx]
        pc_path = info['lidar_path']
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(NuScenes)
