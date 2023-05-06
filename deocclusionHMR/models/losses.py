import torch
import torch.nn as nn

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1]))
        loss = loss.sum(dim=(1, 2))
        return loss


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:,  :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss

class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-2)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_temp = self.loss_fn(pred_param, gt_param).sum(1)
        loss_param = (has_param * loss_temp)
        return loss_param

class BoneLoss(nn.Module):
    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(BoneLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, 14,64,64]
            gt_param (torch.Tensor): Tensor of shape [B, 14, 64,64]
            has_param:某些通道不存在 不计算损失
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]

        # num_dims = len(pred_param.shape)
        # mask_dimension = [batch_size] + [1] * (num_dims-2)
        # has_param = has_param.type(pred_param.type()).view(*mask_dimension)

        loss_temp = self.loss_fn(pred_param, gt_param)
        loss_temp = loss_temp.sum(dim=(2,3))
        loss_param = (has_param * loss_temp).sum(dim=1)
        return loss_param