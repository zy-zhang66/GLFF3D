import MinkowskiEngine as ME

from mmdet.models import DETECTORS
from mmdet3d.models import build_backbone, build_head, builder
from mmdet3d.core import bbox3d2result

import torch

from .base import Base3DDetector

from torch import nn
import torch.nn.functional as F

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# convolutional channel mixer (CC)
class CC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.CC = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.CC(x)


# spatially-adaptive feature modulation (SA)
class SA(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out
class GMFE(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.SA = SA(dim)
        self.CC = CC(dim, ffn_scale)

    def forward(self, x):
        x = self.SA(self.norm1(x)) + x
        x = self.CC(self.norm2(x)) + x
        return x
@DETECTORS.register_module()
class SingleStageSparse3DDetector(Base3DDetector):
    def __init__(self,
                 backbone,
                 neck_with_head,
                 voxel_size,
                 pretrained=False,
                 train_cfg=None,
                 test_cfg=None):
        super(SingleStageSparse3DDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        neck_with_head.update(train_cfg=train_cfg)
        neck_with_head.update(test_cfg=test_cfg)
        self.neck_with_head = build_head(neck_with_head)
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()


    def init_weights(self, pretrained=None):
        self.backbone.init_weights()
        self.neck_with_head.init_weights()

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:] / 255.) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        x = self.neck_with_head(x)
        return x

    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas):
        x = self.extract_feat(points, img_metas)
        losses = self.neck_with_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        bbox_list = self.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass


@DETECTORS.register_module()
class SingleStageSparse3DDetector_CA(Base3DDetector):
    def __init__(self,
                 backbone,
                 neck_with_head,
                 voxel_size,
                 stage2_head=None,
                 img_backbone=None,
                 img_neck=None,
                 img_encoder=None,
                 freeze_img_branch=False,
                 freeze_stage1=False,
                 pretrained=False,
                 train_cfg=None,
                 test_cfg=None):
        super(SingleStageSparse3DDetector_CA, self).__init__()
        self.backbone = build_backbone(backbone)
        neck_with_head.update(train_cfg=train_cfg)
        neck_with_head.update(test_cfg=test_cfg)
        self.neck_with_head = build_head(neck_with_head)
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # Initialize GFME modules for each scale
        self.GFME1 = GFME(dim=256)  # Assuming the feature dimension is 256, adjust as necessary
        self.GFME2 = GFME(dim=256)
        self.GFME3 = GFME(dim=256)
        self.GFME4 = GFME(dim=256)

        self.fuse_img_features = None  

        # image branch
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_encoder is not None:
            self.img_encoder = builder.build_head(img_encoder)

        self.freeze_img_branch = freeze_img_branch
        if freeze_img_branch:
            self.freeze_img_branch_params()
        if freeze_stage1:
            self.freeze_stage1_params()
        if stage2_head is not None:
            self.stage2_head = build_head(stage2_head)

    def freeze_stage1_params(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.neck_with_head.parameters():
            param.requires_grad = False

    def freeze_img_branch_params(self):
        """Freeze all image branch parameters."""
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:] / 255.) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)

        x = self.backbone(x)
              
        x, select_points = self.neck_with_head(x)
       
        return x, select_points

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Overload in order to load img network ckpts into img branch."""
        for key in list(state_dict):
            if not key.startswith('img_bbox_head'):
                continue

            if 'encoder' in key or 'level_embeds' in key:
                key_ = key.replace('img_bbox_head.transformer',
                                   'img_encoder')
                state_dict[key_] = state_dict.pop(key)
            else:
                state_dict.pop(key)

        super(Base3DDetector, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                          strict, missing_keys, unexpected_keys,
                                                          error_msgs)

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @torch.no_grad()
    def extract_img_feat(self, img, img_metas):
        """Directly extract features from the img backbone+neck."""
        x = self.img_backbone(img)


       
        if self.with_img_neck:
            x = self.img_neck(x)
        #
        x = list(x)  
        # Apply GFME to each feature scale
        x[0] = self.GFME1(x[0])  # (8, 256, 100, 140)
        x[1] = self.GFME2(x[1])  # (8, 256, 50, 70)
        x[2] = self.GFME3(x[2])  # (8, 256, 25, 35)
        x[3] = self.GFME4(x[3])  # (8, 256, 13, 18)
        x = tuple(x)
        

        if self.img_encoder:
            x = self.img_encoder(x, img_metas)
        return x


    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      img=None):
        # 
        x, select_points = self.extract_feat(points, img_metas)
        

        #
        losses, targets = self.neck_with_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)

        
        if self.with_img_backbone:
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape

            
            img_features = self.extract_img_feat(img, img_metas)
            
            img_dict = dict(
                img_features=img_features,
                img_metas=img_metas,
            )
        else:
            img_dict = None
       
        stage2_preds = self.stage2_head(select_points, img_dict)
        

        
        stage2_targets = self.get_stage2_targets(targets, select_points[-1])
        

        
        stage2_losses = self.stage2_head.loss(stage2_preds, stage2_targets, select_points[0])
        

        losses.update(stage2_losses)
        

        return losses

    def get_stage2_targets(self, targets, sort_inds):
        '''
        targets[List(List)]: out length batch size, in length 3 which means centerness_targets, bbox_targets and labels
        returns:
            stage2_targets[List]: lengths which composed of selected centerness_targets, bbox_targets and labels
            centerness_targets shape batch * N, bbox_targets shape batch * N * 8, labels shape batch * N
            N is the number of points
        '''
        batch_size = len(targets)
        centerness_targets, bbox_targets, labels = [], [], []
        for i in range(batch_size):
            targets_per_img, sort_inds_per_img = targets[i], sort_inds[i]
            centerness_targets.append(targets_per_img[0][sort_inds_per_img])
            bbox_targets.append(targets_per_img[1][sort_inds_per_img])
            labels.append(targets_per_img[2][sort_inds_per_img])
        centerness_targets = torch.stack(centerness_targets, dim=0)
        bbox_targets = torch.stack(bbox_targets, dim=0)
        labels = torch.stack(labels, dim=0)
        return (centerness_targets, bbox_targets, labels)

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        x, select_points = self.extract_feat(points, img_metas)
        stage1_results = self.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)
        # img branch
        if self.with_img_backbone:
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
            img_features = self.extract_img_feat(img, img_metas)
            img_dict = dict(
                img_features=img_features,
                img_metas=img_metas,
            )
        else:
            img_dict = None
        stage2_preds = self.stage2_head(select_points, img_dict)
        stage2_results = self.stage2_head.get_bboxes(stage2_preds, select_points[0], img_metas, rescale=rescale)
        bbox_list = []
        # per image
        for i in range(len(stage2_results)):
            results = [stage1_results[i], stage2_results[i]]
            ensemble_bboxes, ensemble_scores = [], []
            for stage in self.test_cfg.ensemble_stages:
                ensemble_bboxes.append(results[stage][0])
                ensemble_scores.append(results[stage][1])
            ensemble_bboxes = torch.cat(ensemble_bboxes, dim=0)
            ensemble_scores = torch.cat(ensemble_scores, dim=0)
            nms_process = self.neck_with_head._nms(ensemble_bboxes, ensemble_scores, img_metas[i])
            bbox_list.append(nms_process)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass


@DETECTORS.register_module()
class TwoStageSparse3DDetector(SingleStageSparse3DDetector_CA):
    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        x, select_points = self.extract_feat(points, img_metas)
        stage1_results = self.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)
        # img branch
        if self.with_img_backbone:
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
            img_features = self.extract_img_feat(img, img_metas)
            img_dict = dict(
                img_features=img_features,
                img_metas=img_metas,
            )
        else:
            img_dict = None
        stage2_preds = self.stage2_head(select_points, img_dict)
        stage2_results_all = self.stage2_head.get_bboxes(stage2_preds, select_points[0], img_metas, rescale=rescale)

        bbox_list = []
        # per image
        for i in range(len(stage1_results)):
            results = [stage1_results[i]]
            for stage2_results in stage2_results_all:
                results.append(stage2_results[i])
            ensemble_bboxes, ensemble_scores = [], []
            for stage in self.test_cfg.ensemble_stages:
                ensemble_bboxes.append(results[stage][0])
                ensemble_scores.append(results[stage][1])
            ensemble_bboxes = torch.cat(ensemble_bboxes, dim=0)
            ensemble_scores = torch.cat(ensemble_scores, dim=0)
            nms_process = self.neck_with_head._nms(ensemble_bboxes, ensemble_scores, img_metas[i])
            bbox_list.append(nms_process)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results


@DETECTORS.register_module()
class TwoStageSparse3DDetectorFaster(Base3DDetector):
    def __init__(self,
                 backbone,
                 neck_with_head,
                 voxel_size,
                 stage2_head=None,
                 img_backbone=None,
                 img_neck=None,
                 img_encoder=None,
                 freeze_img_branch=False,
                 freeze_stage1=False,
                 pretrained=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        neck_with_head.update(train_cfg=train_cfg)
        neck_with_head.update(test_cfg=test_cfg)
        self.neck_with_head = build_head(neck_with_head)
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # image branch
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)

        self.freeze_img_branch = freeze_img_branch
        if freeze_img_branch:
            self.freeze_img_branch_params()
        if freeze_stage1:
            self.freeze_stage1_params()
        if stage2_head is not None:
            self.stage2_head = build_head(stage2_head)

    def freeze_stage1_params(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.neck_with_head.parameters():
            param.requires_grad = False

    def freeze_img_branch_params(self):
        """Freeze all image branch parameters."""
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:] / 255.) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        x, select_points = self.neck_with_head(x)
        return x, select_points

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Overload in order to load img network ckpts into img branch."""
        module_names = ['backbone', 'neck']
        for key in list(state_dict):
            for module_name in module_names:
                if key.startswith(module_name) and ('img_' +
                                                    key) not in state_dict:
                    state_dict['img_' + key] = state_dict.pop(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @torch.no_grad()
    def extract_img_feat(self, img, img_metas):
        """Directly extract features from the img backbone+neck."""
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        return x

    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      img=None):
        x, select_points = self.extract_feat(points, img_metas)

        #

        losses, targets = self.neck_with_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        # img branch
        if self.with_img_backbone:
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
            img_features = self.extract_img_feat(img, img_metas)
            img_dict = dict(
                img_features=img_features,
                img_metas=img_metas,
            )
        else:
            img_dict = None
        stage2_preds = self.stage2_head(select_points, img_dict)
        stage2_targets = self.get_stage2_targets(targets, select_points[-1])
        stage2_losses = self.stage2_head.loss(stage2_preds, stage2_targets, select_points[0])
        losses.update(stage2_losses)
        return losses

    def get_stage2_targets(self, targets, sort_inds):
        batch_size = len(targets)
        centerness_targets, bbox_targets, labels = [], [], []
        for i in range(batch_size):
            targets_per_img, sort_inds_per_img = targets[i], sort_inds[i]
            centerness_targets.append(targets_per_img[0][sort_inds_per_img])
            bbox_targets.append(targets_per_img[1][sort_inds_per_img])
            labels.append(targets_per_img[2][sort_inds_per_img])
        centerness_targets = torch.stack(centerness_targets, dim=0)
        bbox_targets = torch.stack(bbox_targets, dim=0)
        labels = torch.stack(labels, dim=0)
        return (centerness_targets, bbox_targets, labels)

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        x, select_points = self.extract_feat(points, img_metas)
        stage1_results = self.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)
        # img branch
        if self.with_img_backbone:
            batch_input_shape = tuple(img[0].size()[-2:])
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
            img_features = self.extract_img_feat(img, img_metas)
            img_dict = dict(
                img_features=img_features,
                img_metas=img_metas,
            )
        else:
            img_dict = None
        stage2_preds = self.stage2_head(select_points, img_dict)
        stage2_results_all = self.stage2_head.get_bboxes(stage2_preds, select_points[0], img_metas, rescale=rescale)

        bbox_list = []
        # per image
        for i in range(len(stage1_results)):
            results = [stage1_results[i]]
            for stage2_results in stage2_results_all:
                results.append(stage2_results[i])
            ensemble_bboxes, ensemble_scores = [], []
            for stage in self.test_cfg.ensemble_stages:
                ensemble_bboxes.append(results[stage][0])
                ensemble_scores.append(results[stage][1])
            ensemble_bboxes = torch.cat(ensemble_bboxes, dim=0)
            ensemble_scores = torch.cat(ensemble_scores, dim=0)
            nms_process = self.neck_with_head._nms(ensemble_bboxes, ensemble_scores, img_metas[i])
            bbox_list.append(nms_process)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass
