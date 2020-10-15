from abc import ABC, abstractmethod

from torch.utils.data.dataloader import DataLoader
from torch import nn

from anomaly_detection.dpa.layers import FadeinLayer, ConcatLayer, EqualLayer
from anomaly_detection.dpa.pg_networks import ProgGrowStageType
from anomaly_detection.utils.datasets import CIFAR10Dataset, SVHNDataset


class AbstractProgGrowGenerator(ABC):
    @abstractmethod
    def set_stage_resolution(self, stage, resolution, batch_size):
        pass

    @abstractmethod
    def set_progress(self, progress):
        pass

    @abstractmethod
    def __next__(self):
        pass


class ProgGrowImageGenerator(AbstractProgGrowGenerator):
    def __init__(self, dataset, max_resolution, batch_size, inf=False):
        super(ProgGrowImageGenerator, self).__init__()

        self._dataset = dataset
        self._max_resolution = max_resolution
        self._inf = inf
        self._batch_size = batch_size

        if isinstance(self._dataset, CIFAR10Dataset) or isinstance(self._dataset, SVHNDataset):
            self.num_workers = 0
        else:
            self.num_workers = 8

        self._data_loader = DataLoader(self._dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                       num_workers=self.num_workers, pin_memory=False)
        self._image_gen = iter(self._data_loader)

        self._resolution = max_resolution
        self._stage = ProgGrowStageType.stab
        self._mix_res_module = MixResolution(self._stage, self._resolution, self._max_resolution)

    def set_stage_resolution(self, stage, resolution, batch_size):
        self._stage = stage
        self._resolution = resolution

        # create new iterator if it is necessary
        if self._batch_size != batch_size:
            self._batch_size = batch_size
            self._data_loader = DataLoader(self._dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                           num_workers=self.num_workers, pin_memory=False)
            self._image_gen = iter(self._data_loader)

        self._mix_res_module = MixResolution(self._stage, self._resolution, self._max_resolution)

    def set_progress(self, progress):
        self._mix_res_module.set_progress(progress)

    def __iter__(self):
        self._image_gen = iter(self._data_loader)
        return self

    def __len__(self):
        return len(self._data_loader)

    def __next__(self):
        images = next(self._image_gen, None)
        if images is None:
            if self._inf:
                # del self._image_gen
                self._image_gen = iter(self._data_loader)
                images = next(self._image_gen, None)
            else:
                raise StopIteration()

        return self._mix_res_module(images).cuda()


class MixResolution(nn.Module):
    def __init__(self, stage, resolution, max_resolution):
        super(MixResolution, self).__init__()
        self._stage = stage

        if resolution == max_resolution:
            high_res = EqualLayer()
        else:
            scale_factor = int(max_resolution / resolution)
            high_res = nn.Sequential(
                nn.Upsample(scale_factor=1/scale_factor, mode='bilinear'),
            )

        if stage == ProgGrowStageType.stab:
            self.mix_res_model = high_res
        elif stage == ProgGrowStageType.trns:
            self.mix_res_model = nn.Sequential()

            scale_factor = int(max_resolution / (resolution / 2))
            low_res = nn.Sequential(
                nn.Upsample(scale_factor=1/scale_factor, mode='bilinear'),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            self.mix_res_model.add_module('concat', ConcatLayer(low_res, high_res))
            self.mix_res_model.add_module('fadein', FadeinLayer())
        else:
            raise NotImplementedError

    def set_progress(self, progress):
        if self._stage == ProgGrowStageType.trns:
            self.mix_res_model.fadein.set_progress(progress)

    def forward(self, x):
        return self.mix_res_model(x)
