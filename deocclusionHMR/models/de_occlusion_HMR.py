import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from typing import Any, Dict, Tuple

from prohmr.models import SMPL
from yacs.config import CfgNode

from prohmr.utils import SkeletonRenderer
from prohmr.utils.geometry import aa_to_rotmat, perspective_projection
from prohmr.optimization import OptimizationTask

from .backbones import create_backbone, BasicBlock
from .heads import SMPLFlow
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, BoneLoss
from nflows.flows import ConditionalGlow
from prohmr.utils.geometry import rot6d_to_rotmat
from torchvision import transforms


class Map2vec(nn.Module):
    def __init__(self):
        super(Map2vec, self).__init__()
        model = [
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #32
            nn.Conv2d(256+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #16
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #4
        ]
        self.layers = nn.Sequential(*model)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 256 // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256 // 8, 512, kernel_size=1),
            nn.Sigmoid()
        )
        #self.fc = nn.Linear(1024, output_channel)

    def forward(self, x):
        # attention dropout
        #x3 = response.view(x.shape[0], x3.shape[1], 16, 16)
        w = self.se(x)
        x1 = x * w
        x = x + x1

        x = self.layers(x)
        x = x.mean(dim=(2, 3))
        #x = self.fc(x)
        return x


class Segment(nn.Module):
    def __init__(self):
        super(Segment, self).__init__()
        #8->16->32->64
        self.up1 = nn.Upsample(scale_factor=2) #16
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024,affine=True)
        self.relu1 = nn.ReLU()
        self.up2 = nn.Upsample(scale_factor=2)#32
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512, affine=True)
        self.relu2 = nn.ReLU()
        self.up3 = nn.Upsample(scale_factor=2)#64
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.up3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x1 = self.relu3(x)

        x2 = self.conv4(x1)
        x2 = nn.Sigmoid()(x2)

        return x2, x1
        




class FCHead_camera(nn.Module):
    def __init__(self):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(FCHead_camera, self).__init__()

        model = [nn.Linear(2048, 512), nn.ReLU(inplace=False), nn.Linear(512, 3)]
        self.layers = nn.Sequential(*model)


        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load('data/smpl_mean_params.npz')
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32))[ None]
        #init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None]


        self.register_buffer('init_cam', init_cam)
        #self.register_buffer('init_betas', init_betas)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass.
        Args:
            smpl_params (Dict): Dictionary containing predicted SMPL parameters.
            feats (torch.Tensor): Tensor of shape (N, C) containing the features computed by the backbone.
        Returns:
            pred_betas (torch.Tensor): Predicted SMPL betas.
            pred_cam (torch.Tensor): Predicted camera parameters.
        """

        batch_size = feats.shape[0]
        # offset = self.layers(feats).reshape(batch_size, 13)#.repeat(1, num_samples, 1)
        offset = self.layers(feats).reshape(batch_size, 3)
        #betas_offset = offset[:, :10]
        #cam_offset = offset[:, 10:]
        pred_cam = offset + self.init_cam
        #pred_betas = betas_offset + self.init_betas
        # return pred_betas, pred_cam
        return pred_cam


class ProHMR2(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        """
        Setup ProHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)


        self.human_mask = Segment()
        self.occluder_mask = Segment()

        self.img_feature1 = Map2vec()
        self.pose_shape = nn.Sequential(*[nn.Linear(1024, 512),
                                          nn.ReLU(inplace=False),
                                          nn.Linear(512, 144+10)
                                          ])


        # Directly predict shape，camera parameters
        self.camera = FCHead_camera()

        # Create discriminator
        self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()

        # Instantiate SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)                   

        # Setup renderer for visualization
        self.renderer = SkeletonRenderer(self.cfg)

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

        mean_params = np.load('data/smpl_mean_params.npz')
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None]

        self.register_buffer('init_betas', init_betas)







    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) +
                                             list(self.pose_shape.parameters()) +
                                             list(self.img_feature1.parameters()) +
                                             list(self.camera.parameters()) +
                                             list(self.human_mask.parameters()) +
                                             list(self.occluder_mask.parameters()),
                                     lr=self.cfg.TRAIN.LR,
                                     weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                           lr=self.cfg.TRAIN.LR,
                                           weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        return optimizer, optimizer_disc

    def initialize(self, batch: Dict, conditioning_feats: torch.Tensor):
        """
        Initialize ActNorm buffers by running a dummy forward step
        Args:
            batch (Dict): Dictionary containing batch data
            conditioning_feats (torch.Tensor): Tensor of shape (N, C) containing the conditioning features extracted using thee backbonee
        """
        # Get ground truth SMPL params, convert them to 6D and pass them to the flow module together with the conditioning feats.
        # Necessary to initialize ActNorm layers.
        smpl_params = {k: v.clone() for k,v in batch['smpl_params'].items()}
        batch_size = smpl_params['body_pose'].shape[0]
        has_smpl_params = batch['has_smpl_params']['body_pose'] > 0
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)[has_smpl_params]
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)[has_smpl_params]
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)[has_smpl_params]
        conditioning_feats = conditioning_feats[has_smpl_params]
        with torch.no_grad():
            _, _ = self.flow.log_prob(smpl_params, conditioning_feats)
            self.initialized |= True

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        # if train:
        #     x = batch['img']
        # else:
        #     x = batch
        x = batch['img']
        batch_size = x.shape[0]

        # Compute keypoint features using the backbone
        conditioning_feats_human, conditioning_feats_occluder, loss_diver2, loss_sim = self.backbone(x) # 2048

        # human mask
        human_mask, human_features = self.human_mask(conditioning_feats_human)
        occluder_mask, mask_feature = self.occluder_mask(conditioning_feats_occluder)
        conditioning_feats = conditioning_feats_human.mean(dim=(2,3))

        # add layer dropout
        loss_diver1 = torch.tensor(0., device=x.device).float()
        for name, param in self.backbone.layer3.named_parameters():
            if '5.conv2.weight' in name and param.requires_grad and len(param.shape) == 4:
                N, C, H, W = param.shape
                weight = param.view(N, C, H*W)
                for j in range(H*W):
                    # self-corr
                    weight_T = torch.einsum('ij->ji', [weight[:, :, j]])
                    corr = torch.einsum('ij,jm->im', [weight[:, :, j], weight_T])  # 256*1 1*256
                    corr *= (1 - torch.eye(256, 256, device=x.device))

                    # norm matrix
                    norm = torch.norm(weight[:, :, j], dim=1).unsqueeze(1)
                    norm_T = torch.einsum('ij->ji', [norm])
                    norm_matrix = torch.einsum('ij,jm->im', [norm, norm_T])
                    norm_matrix = 1. / (norm_matrix + 1e-6)

                    loss_diver11 = corr * norm_matrix
                    loss_diver11 = torch.abs(loss_diver11)
                    loss_diver1 += loss_diver11.sum()

        loss_diver1 /= (256*256)



        # # ffeature = torch.cat([mask_feature, human_features], dim=1)
        # #if has_occ_mask:
        # occluder_mask_clone = occluder_mask.clone()
        # # if torch.numel(torch.nonzero(has_occ_mask)) > 0:
        # #     occluder_mask_clone[has_occ_mask == 1, 0, :, :] = gt_occluder_mask[has_occ_mask == 1, :, :]

        ffeature = torch.cat([human_features, mask_feature], dim=1)


        # 骨架
        img_feature = self.img_feature1(ffeature) #B*1024*1


        pred_cam = self.camera(conditioning_feats) #B 17 64 64
        pose_shape = self.pose_shape(img_feature)
        rotato = pose_shape[:,:144]
        pred_betas = pose_shape[:, 144:]

        pred_pose = rot6d_to_rotmat(rotato).view(batch_size, 24, 3, 3)
        pred_smpl_params = {'global_orient': pred_pose[:, [0]], 'body_pose': pred_pose[:, 1:]}
        pred_smpl_params['betas'] = pred_betas

        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
        output['pred_pose_6d'] = rotato.unsqueeze(1)

        output['loss_div1'] = loss_diver1
        output['loss_div2'] = loss_diver2
        output['loss_sim'] = loss_sim
        output['mask'] = human_mask
        output['occluder'] = occluder_mask

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t

        # Compute model vertices, joints and the projected joints
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices

        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        #print(pred_keypoints_2d.shape)
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)

        if train:
            return output
        else:
            output['faces'] = self.smpl.faces

        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_smpl_params = output['pred_smpl_params']
        pred_pose_6d = output['pred_pose_6d']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']


        batch_size = pred_smpl_params['body_pose'].shape[0]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']
        gt_keypoints_openpose = batch['openpose']

        #mask
        if train:
            has_human_mask = batch['has_mask']
            has_occ_mask = batch['has_occ_mask']
            gt_human_mask = batch['masks']
            gt_occluder_mask = batch['occ_masks']
            gt_human_mask = transforms.functional.resize(gt_human_mask, [64, 64], interpolation=0)
            gt_occluder_mask = transforms.functional.resize(gt_occluder_mask, [64, 64], interpolation=0)


            pred_mask = output['mask'].squeeze(1)
            pred_occluder = output['occluder'].squeeze(1)

            loss_human = nn.MSELoss(reduction='none')(gt_human_mask, pred_mask)
            loss_human = loss_human.mean(dim=(1, 2))
            loss_mask_human = has_human_mask * loss_human

            #loss_mask_human = nn.MSELoss()(gt_human_mask, pred_mask)
            #gt_mask_bk = 1. - gt_mask
            loss_mask_bk = has_occ_mask * (nn.MSELoss(reduction='none')(gt_occluder_mask, pred_occluder)).mean(dim=(1, 2))

            loss_mask_human = loss_mask_human.sum() / batch_size
            loss_mask_bk = loss_mask_bk.sum() / batch_size

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)

        loss_keypoints_openpose = self.keypoint_2d_loss(pred_keypoints_2d[:, 0:25, :], gt_keypoints_openpose)
        loss_keypoints_2d = loss_keypoints_2d + loss_keypoints_openpose

        # Compute loss on SMPL parameters--level3
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].unsqueeze(1).view(batch_size , -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_smpl_params[k].unsqueeze(1)
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)
        loss_smpl_params_mode = {k: v.sum() / batch_size for k, v in loss_smpl_params.items()}


        # Compute mode and expectation losses for 3D and 2D keypoints
        # The first item of the second dimension always corresponds to the mode
        loss_keypoints_2d_mode = loss_keypoints_2d.sum() / batch_size
        loss_keypoints_3d_mode = loss_keypoints_3d.sum() / batch_size

        # Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d = loss_pose_6d.reshape(batch_size, -1)
        loss_pose_6d_mode = loss_pose_6d.mean()

        if train:
            loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode+\
                   self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_MODE'] * loss_keypoints_2d_mode+\
                   self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * (loss_pose_6d_mode) + \
                   output['loss_div1'] + output['loss_div2'] + output['loss_sim'] + \
                   loss_mask_human + loss_mask_bk +\
                   sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k + '_MODE').upper()] for k in loss_smpl_params_mode])

            losses = dict(loss=loss.detach(),
                          loss_keypoints_2d_mode=loss_keypoints_2d_mode.detach(),
                          loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach(),
                          loss_pose_6d_mode=loss_pose_6d_mode.detach(),
                          loss_diver1 = output['loss_div1'].detach(),
                          loss_diver2 = output['loss_div2'].detach(),
                          loss_sim = output['loss_sim'].detach(),
                          loss_mask_human = loss_mask_human.detach(),
                          loss_mask_bk = loss_mask_bk.detach()
                          )
        else:
            loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode + \
                   self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_MODE'] * loss_keypoints_2d_mode + \
                   self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * (loss_pose_6d_mode) + \
                   output['loss_div1'] + output['loss_div2'] + output['loss_sim'] + \
                   sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k + '_MODE').upper()] for k in
                        loss_smpl_params_mode])

            losses = dict(loss=loss.detach(),
                          loss_keypoints_2d_mode=loss_keypoints_2d_mode.detach(),
                          loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach(),
                          loss_pose_6d_mode=loss_pose_6d_mode.detach(),
                          loss_diver1=output['loss_div1'].detach(),
                          loss_diver2=output['loss_div2'].detach(),
                          loss_sim=output['loss_sim'].detach()
                          )

        for k, v in loss_smpl_params_mode.items():
            losses['loss_' + k + '_mode'] = v.detach()

        output['losses'] = losses

        return loss

    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        summary_writer = self.logger.experiment
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        images = 255*images.permute(0, 2, 3, 1).cpu().numpy()
        num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES if mode == 'train' else self.cfg.TRAIN.NUM_TEST_SAMPLES

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, num_samples, -1, 3)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, num_samples, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, num_samples, -1, 2)

        for loss_name, val in losses.items():
            summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)
        num_samples_per_image = min(num_samples, self.cfg.EXTRA.NUM_LOG_SAMPLES_PER_IMAGE)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, num_samples, -1, 3)

        pred_human_mask = output['mask'].detach()
        pred_occ_mask = output['occluder'].detach()

        # if train:
        #     gt_mask = batch['masks'].unsqueeze(1)
        #     occ_mask = batch['occ_masks'].unsqueeze(1)
        #     # from torchvision.utils import save_image
        #     # path = '/home/qihangran/bone1_.jpg'
        #     # path2 = '/home/qihangran/bone2_.jpg'
        #     # save_image(gt_mask[0], path)
        #     #save_image(torch.from_numpy(mask_path).unsqueeze(0), path2)
        #     #save_image(torch.from_numpy(img_patch) / 255.0, '/home/hangran/oooooooooooooooooooo.jpg')
        #
        #     # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #     predictions = self.renderer(pred_keypoints_3d[:num_images, :num_samples_per_image],
        #                                 gt_keypoints_3d[:num_images],
        #                                 2 * gt_keypoints_2d[:num_images],
        #                                 images=images[:num_images],
        #                                 camera_translation=pred_cam_t[:num_images, :num_samples_per_image],
        #                                 human_mask=gt_mask[:num_images],
        #                                 occ_mask = occ_mask[:num_images])
        #     summary_writer.add_image('%s/predictions' % mode, predictions.transpose((2, 0, 1)), step_count)
        # else:
        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        predictions = self.renderer(pred_keypoints_3d[:num_images, :num_samples_per_image],
                                    gt_keypoints_3d[:num_images],
                                    2 * gt_keypoints_2d[:num_images],
                                    images=images[:num_images],
                                    camera_translation=pred_cam_t[:num_images, :num_samples_per_image],
                                    human_mask=pred_human_mask[:num_images],
                                    occ_mask = pred_occ_mask[:num_images])
        summary_writer.add_image('%s/predictions' % mode, predictions.transpose((2, 0, 1)), step_count)


    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer, optimizer_disc = self.optimizers(use_pl_optimizer=True)
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        #num_samples = pred_smpl_params['body_pose'].shape[1]
        num_samples = 1
        #pred_smpl_params = output['pred_smpl_params']
        loss = self.compute_loss(batch, output, train=True)
        disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1), pred_smpl_params['betas'].reshape(batch_size * num_samples, -1))
        loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
        loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1), pred_smpl_params['betas'].reshape(batch_size * num_samples, -1), optimizer_disc)
        output['losses']['loss_gen'] = loss_adv
        output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)
        #self.tensorboard_logging(batch, output, self.global_step, train=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        pred_smpl_params = output['pred_smpl_params']
        #num_samples = pred_smpl_params['body_pose'].shape[1]
        num_samples = 1
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

    def downstream_optimization(self, regression_output: Dict, batch: Dict, opt_task: OptimizationTask, **kwargs: Any) -> Dict:
        """
        Run downstream optimization using current regression output
        Args:
            regression_output (Dict): Dictionary containing batch data
            batch (Dict): Dictionary containing batch data
            opt_task (OptimizationTask): Class object for desired optimization task. Must implement __call__ method.
        Returns:
            Dict: Dictionary containing regression output.
        """
        conditioning_feats = regression_output['conditioning_feats']
        flow_net = lambda x: self.flow(conditioning_feats, z=x)
        return opt_task(flow_net=flow_net,
                        regression_output=regression_output,
                        data=batch,
                        **kwargs)
