import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        # print(batch_dict.keys())# dict_keys(['points', 'frame_id', 'metadata', 'gt_boxes', 'num_sampled_points',
        # 'visibility', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'batch_size', 'pillar_features'])
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1  # 4
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            # print('spatial-feature',spatial_feature.shape)#torch.Size([64, 262144])
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                             self.nx)
        # batch_dict['spatial_features'] = batch_spatial_features
        '''
        fusion visibility
        '''
        visibility = batch_dict['visibility']
        # print('visibility:', visibility.shape)  # torch.Size([4, 512, 512])
        # print('batch:', batch_spatial_features.shape)  # torch.Size([4, 64, 512, 512])
        fusion_features = []
        for batch_idx in range(batch_size):
            fusion = batch_spatial_features[batch_idx] * visibility[batch_idx]
            fusion_features.append(fusion)

        fusion_features = torch.stack(fusion_features, 0)
        batch_dict['spatial_features'] = fusion_features
        # print('fusion_features:', fusion_features.shape)#torch.Size([4, 64, 512, 512])
        '''
        fusion visibility
        '''
        return batch_dict
