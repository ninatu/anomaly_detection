from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod

from torch import nn


class ProgGrowStageType(Enum):
    trns = 'trns'  # translation stage - increasing resolution twice
    stab = 'stab'  # stabilization stage - training at a fixed resolution


class NetworkType(Enum):
    regular = 'regular'
    residual9 = 'residual9'
    residual18 = 'residual18'


class AbstractNetwork(nn.Module, ABC):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_progress(self):
        pass

    @abstractmethod
    def set_progress(self, progress):
        pass

    def forward(self, x):
        return self.model(x)


class STABNetwork(AbstractNetwork):
    def __init__(self, model):
        super().__init__(model)

    def get_progress(self):
        return 1

    def set_progress(self, progress):
        pass


class TRNSNetwork(AbstractNetwork):
    def __init__(self, model):
        super().__init__(model)
        try:
            model.fadein
        except Exception:
            raise ValueError('Model have to have Fadein layer, called "fadein"')

    def get_progress(self):
        return self.model.fadein.get_progress()

    def set_progress(self, progress):
        self.model.fadein.set_progress(progress)


class ProgGrowNetworks(nn.Module):
    def __init__(self):
        super(ProgGrowNetworks, self).__init__()
        self._resolution_nets_dict = defaultdict(lambda: dict())

    def get_net(self, stage, resolution):
        return self._resolution_nets_dict[resolution][stage]

    def set_net(self, stage, resolution, network):
        if stage == ProgGrowStageType.trns:
            assert isinstance(network, TRNSNetwork)
        elif stage == ProgGrowStageType.stab:
            assert isinstance(network, STABNetwork)
        else:
            raise ValueError("Stage: {} is not avaliable".format(stage))
        self._resolution_nets_dict[resolution][stage] = network
