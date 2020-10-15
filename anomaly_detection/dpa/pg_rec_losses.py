import torch
from abc import ABC, abstractmethod

from anomaly_detection.dpa.rec_losses import L2Loss, L1Loss, ReconstructionLossType, PerceptualLoss
from anomaly_detection.dpa.pg_networks import ProgGrowStageType


class AbstractPGLoss(torch.nn.Module, ABC):
    def __init__(self, max_resolution):
        super().__init__()

        self._resolution = max_resolution
        self._stage = ProgGrowStageType.stab
        self._progress = 0

    @abstractmethod
    def set_stage_resolution(self, stage, resolution):
        pass

    @abstractmethod
    def set_progress(self, progress):
        pass

    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def set_reduction(self, reduction):
        pass


class PGPerceptualLoss(AbstractPGLoss):
    def __init__(self, max_resolution, weights_per_resolution,
                 reduction='mean',
                 use_smooth_pg=False,
                 use_feature_normalization=False,
                 use_L1_norm=False,
                 use_relative_error=False):
        super(PGPerceptualLoss, self).__init__(max_resolution)

        self._max_resolution = max_resolution
        self._weights_per_resolution = weights_per_resolution
        self._use_smooth_pg = use_smooth_pg
        self._loss = PerceptualLoss(reduction=reduction,
                                    use_feature_normalization=use_feature_normalization,
                                    use_L1_norm=use_L1_norm,
                                    use_relative_error=use_relative_error)

        self._resolution = self._max_resolution
        self._stage = ProgGrowStageType.stab
        self._progress = 0

    def set_stage_resolution(self, stage, resolution):
        self._stage = stage
        self._resolution = resolution
        self._progress = 0

    def set_progress(self, progress):
        self._progress = progress

    def set_reduction(self, reduction):
        self._loss.reduction = reduction

    def forward(self, x, y):
        self._loss.set_new_weights(**self._weights_per_resolution[self._resolution])
        loss = self._loss(x, y)

        if self._use_smooth_pg:
            if self._stage == ProgGrowStageType.trns and self._progress < 1:
                prev_res = int(self._resolution / 2)
                self._loss.set_new_weights(**self._weights_per_resolution[prev_res])

                x = torch.nn.functional.upsample(x, scale_factor=0.5, mode='bilinear')
                y = torch.nn.functional.upsample(y, scale_factor=0.5, mode='bilinear')

                prev_loss = self._loss(x, y)
                loss = (1 - self._progress) * prev_loss + self._progress * loss

        return loss


class PGRelativePerceptualL1Loss(PGPerceptualLoss):
    def __init__(self, max_resolution, weights_per_resolution, reduction='mean', use_smooth_pg=False):
        super().__init__(
            max_resolution, weights_per_resolution,
            reduction=reduction,
            use_smooth_pg=use_smooth_pg,
            use_feature_normalization=True,
            use_L1_norm=True,
            use_relative_error=True)


class PGL2Loss(AbstractPGLoss):
    def __init__(self, max_resolution, reduction='mean'):
        super().__init__(max_resolution)
        self._loss = L2Loss(reduction=reduction)

    def set_stage_resolution(self, stage, resolution):
        pass

    def set_progress(self, progress):
        pass

    def set_reduction(self, reduction):
        self._loss.set_reduction(reduction)

    def forward(self, x, y):
        return self._loss(x, y)


class PGL1Loss(AbstractPGLoss):
    def __init__(self, max_resolution, reduction='mean'):
        super().__init__(max_resolution)
        self._loss = L1Loss(reduction=reduction)

    def set_stage_resolution(self, stage, resolution):
        pass

    def set_progress(self, progress):
        pass

    def set_reduction(self, reduction):
        self._loss.set_reduction(reduction)

    def forward(self, x, y):
        return self._loss(x, y)


class PGComposeLoss(AbstractPGLoss):
    def __init__(self, max_resolution, loss_1, loss_2):
        super().__init__(max_resolution)

        self.loss_1 = PG_RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_1['loss_type']]](
            max_resolution=max_resolution, **loss_1['loss_kwargs'])
        self.loss_1_weight = loss_1['loss_weight']

        self.loss_2 = PG_RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_2['loss_type']]](
            max_resolution=max_resolution, **loss_2['loss_kwargs'])
        self.loss_2_weight = loss_2['loss_weight']

    def set_stage_resolution(self, stage, resolution):
        self.loss_1.set_stage_resolution(stage, resolution)
        self.loss_2.set_stage_resolution(stage, resolution)

    def set_progress(self, progress):
        self.loss_1.set_progress(progress)
        self.loss_2.set_progress(progress)

    def forward(self, x, y):
        return self.loss_1_weight * self.loss_1(x, y) + self.loss_2_weight * self.loss_2(x, y)

    def set_reduction(self, reduction):
        self.loss_1.set_reduction(reduction)
        self.loss_2.set_reduction(reduction)


PG_RECONSTRUCTION_LOSSES = {
    ReconstructionLossType.perceptual: PGPerceptualLoss,
    ReconstructionLossType.relative_perceptual_L1: PGRelativePerceptualL1Loss,
    ReconstructionLossType.l1: PGL1Loss,
    ReconstructionLossType.l2: PGL2Loss,
    ReconstructionLossType.compose: PGComposeLoss
}
