import os
from collections import defaultdict, OrderedDict
from enum import Enum
import torch
from torch import nn
from torch.nn import functional as F

from anomaly_detection.dpa.feature_extractor import PretrainedVGG19FeatureExtractor


class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def forward(self, x, y):
        assert len(x.shape) == 4
        losses = ((x - y) * (x - y)).sum(3).sum(2).sum(1) / (x.size(1) * x.size(2) * x.size(3))
        if self._reduction == 'none':
            return losses
        elif self._reduction == 'mean':
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class L1Loss(nn.Module):
    def __init__(self, reduction='none'):
        super(L1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean', 'pixelwise']
        self._reduction = reduction

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean', 'pixelwise']
        self._reduction = reduction

    def forward(self, x, y):
        assert len(x.shape) == 4
        losses = torch.abs(x - y)
        if self._reduction == 'pixelwise':
            return losses

        losses = losses.sum(3).sum(2).sum(1) / (losses.size(1) * losses.size(2) * losses.size(3))
        if self._reduction == 'none':
            return losses
        elif self._reduction == 'mean':
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class PerceptualLoss(torch.nn.Module):
    def __init__(self,
                 reduction='mean',
                 img_weight=0,
                 feature_weights=None,
                 use_feature_normalization=False,
                 use_L1_norm=False,
                 use_relative_error=False):
        super(PerceptualLoss, self).__init__()
        """
        We assume that input is normalized with 0.5 mean and 0.5 std
        """

        assert reduction in ['none', 'sum', 'mean', 'pixelwise']

        MEAN_VAR_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'vgg19_ILSVRC2012_object_detection_mean_var.pt')

        self.vgg19_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.vgg19_std = torch.Tensor([0.229, 0.224, 0.225])

        if use_feature_normalization:
            self.mean_var_dict = torch.load(MEAN_VAR_ROOT)
        else:
            self.mean_var_dict = defaultdict(
                lambda: (torch.tensor([0.0], requires_grad=False), torch.tensor([1.0], requires_grad=False))
            )

        self.reduction = reduction
        self.use_L1_norm = use_L1_norm
        self.use_relative_error = use_relative_error

        self.model = PretrainedVGG19FeatureExtractor()

        self.set_new_weights(img_weight, feature_weights)

    def set_reduction(self, reduction):
        self.reduction = reduction

    def forward(self, x, y):
        # pixel-wise prediction is implemented only if loss is obtained from one layer of vgg
        if self.reduction == 'pixelwise':
            assert (len(self.feature_weights) + (self.img_weight != 0)) == 1

        layers = list(self.feature_weights.keys())
        weights = list(self.feature_weights.values())

        x = self._preprocess(x)
        y = self._preprocess(y)

        f_x = self.model(x, layers)
        f_y = self.model(y, layers)

        loss = None

        if self.img_weight != 0:
            loss = self.img_weight * self._loss(x, y)

        for i in range(len(f_x)):
            # put mean, var on right device
            mean, var = self.mean_var_dict[layers[i]]
            mean, var = mean.to(f_x[i].device), var.to(f_x[i].device)
            self.mean_var_dict[layers[i]] = (mean, var)

            # compute loss
            norm_f_x_val = (f_x[i] - mean) / var
            norm_f_y_val = (f_y[i] - mean) / var

            cur_loss = self._loss(norm_f_x_val, norm_f_y_val)

            if loss is None:
                loss = weights[i] * cur_loss
            else:
                loss += weights[i] * cur_loss

        loss /= (self.img_weight + sum(weights))

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'pixelwise':
            loss = loss.unsqueeze(1)
            scale_h = x.shape[2] / loss.shape[2]
            scale_w = x.shape[3] / loss.shape[3]
            loss = F.interpolate(loss, scale_factor=(scale_h, scale_w), mode='bilinear')
            return loss
        else:
            raise NotImplementedError('Not implemented reduction: {:s}'.format(self.reduction))

    def set_new_weights(self, img_weight=0, feature_weights=None):
        self.img_weight = img_weight
        if feature_weights is None:
            self.feature_weights = OrderedDict({})
        else:
            self.feature_weights = OrderedDict(feature_weights)

    def _preprocess(self, x):
        assert len(x.shape) == 4

        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)

        # denormalize
        vector = torch.Tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1).to(x.device)
        x = x * vector + vector

        # normalize
        x = (x - self.vgg19_mean.reshape(1, 3, 1, 1).to(x.device)) / self.vgg19_std.reshape(1, 3, 1, 1).to(x.device)
        return x

    def _loss(self, x, y):
        if self.use_L1_norm:
            norm = lambda z: torch.abs(z)
        else:
            norm = lambda z: z * z

        diff = (x - y)

        if not self.use_relative_error:
            loss = norm(diff)
        else:
            means = norm(x).mean(3).mean(2).mean(1)
            means = means.detach()
            loss = norm(diff) / means.reshape((means.size(0), 1, 1, 1))

        # perform reduction
        if self.reduction == 'pixelwise':
            return loss.mean(1)
        else:
            return loss.mean(3).mean(2).mean(1)


class RelativePerceptualL1Loss(PerceptualLoss):
    def __init__(self, reduction='mean', img_weight=0, feature_weights=None):
        super().__init__(
            reduction=reduction,
            img_weight=img_weight,
            feature_weights=feature_weights,
            use_feature_normalization=True,
            use_L1_norm=True,
            use_relative_error=True,)


class ReconstructionLossType(Enum):
    perceptual = 'perceptual'
    relative_perceptual_L1 = 'relative_perceptual_L1'
    l1 = 'l1'
    l2 = 'l2'
    compose = 'compose'


RECONSTRUCTION_LOSSES = {
    ReconstructionLossType.perceptual: PerceptualLoss,
    ReconstructionLossType.relative_perceptual_L1: RelativePerceptualL1Loss,
    ReconstructionLossType.l1: L1Loss,
    ReconstructionLossType.l2: L2Loss
}
