from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler

'''
add
'''
from ..mapping import mapping
import torch
from torch import nn
import cv2
from ..simplevis import nuscene_vis

'''
add
'''


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        """
         add raycasting
        """
        points = data_dict['points']
        num_sampled_points = data_dict['num_sampled_points']
        sampled_points, original_points = points[:num_sampled_points], points[num_sampled_points:]
        sampled_points, original_points = np.delete(sampled_points, 3, axis=1), np.delete(original_points, 3,
                                                                                          axis=1)  # delete col 4:intensity
        sensor_origins = np.array([0.0, 0.0, 0.0])
        pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        voxel_size = np.array([0.2, 0.2, 8.0])
        indices=data_dict['indices']
        time_stamps = points[indices[:-1], -1]
        time_stamps = (time_stamps[:-1] + time_stamps[1:]) / 2
        time_stamps = [-1000.0] + time_stamps.tolist() + [1000.0]  # add boundaries
        time_stamps = np.array(time_stamps)
        #print('timestamps', time_stamps.shape)#(11,)
        #logodds, original_mask, sampled_mask = mapping.compute_logodds_and_masks_nuscenes(original_points,
                                                                                          #sampled_points,
                                                                                          #sensor_origins, time_stamps,
                                                                                          #pc_range, min(voxel_size))
        logodds, original_mask, sampled_mask = mapping.compute_logodds_and_masks_no_timestamp(original_points,
                                                                                              sampled_points,
                                                                                              sensor_origins, pc_range,
                                                                                              min(voxel_size))
        #print('log:', logodds.shape)  # (10485760,)(512*512*40)
        occupancy = torch.sigmoid(torch.from_numpy(logodds))  # (occupied:0.7,unknown:0.5,free:0.4)
        occupancy = occupancy.reshape((-1, 40, 512, 512))
        # print('occ',occupancy)
        filter = nn.Conv2d(in_channels=40,
                           out_channels=1,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           dilation=1,
                           groups=1,
                           bias=True)
        visibility = filter(occupancy)
        visibility = visibility.detach().numpy()
        visibility = np.squeeze(visibility)
        data_dict['visibility'] = visibility
        # print('vis:', visibility.shape)  # (512, 512)
        # print(visibility)
        #bev_map = nuscene_vis(points)
        #cv2.imshow('bev', bev_map)
        #cv2.imshow('vis', abs(visibility))
        #cv2.waitKey(0)
        """
         add raycasting
        """
        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
