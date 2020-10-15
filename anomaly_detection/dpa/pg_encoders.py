import math
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn

from anomaly_detection.dpa.pg_networks import ProgGrowNetworks, ProgGrowStageType, \
    STABNetwork, TRNSNetwork, NetworkType
from anomaly_detection.dpa.layers import PreActResnetBlockDown, ConvBlock, PreActResnetBlock, get_act_layer, \
    get_pool_layer, MinibatchStdDevLayer, FadeinLayer, ConcatLayer


class AbstactEncoderNetworks(ProgGrowNetworks, ABC):
    def __init__(self, max_input_res, output_res, input_dim, output_dim, inner_dims,
                 use_wscale=False,
                 use_mbstddev=False,
                 norm='none',
                 pad_type='zero',
                 pool='avg'):
        """
        API allows create Discriminator which output is tensor (for example in case PatchGAN).
        For example, output tensor may be 16x4x4 (output_res = 4, output_dim=16).
        But output_res must be degree of 2(except for 2): 1, 4, 8, 16, 32, ...

        inner_dims -- list of depths of inner convolution layers.

        """
        super().__init__()

        self.max_input_res = max_input_res
        self.output_res = output_res

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_dims = inner_dims

        self.use_wscale = use_wscale
        self.use_mbstddev = use_mbstddev
        self.norm = norm
        self.pad_type = pad_type
        self.pool = pool

        self._create_networks()

    def _create_networks(self):
        res_blocks = []

        # next resolution
        resolution = self._get_last_block_resolution()
        res_blocks += [('res_{}'.format(resolution), self._get_last_block())]
        prev_preprocess = self._get_rgb_block(resolution)

        layers = OrderedDict([('preprocess_res_{}'.format(resolution), prev_preprocess)] + res_blocks)
        stab_model = nn.Sequential(layers)
        self.set_net(ProgGrowStageType.stab, resolution, STABNetwork(stab_model))

        resolution *= 2
        while resolution <= self.max_input_res:
            # trns network

            low_res = nn.Sequential(OrderedDict([
                ('from_res_{}_to_res_{}'.format(resolution, resolution // 2), nn.AvgPool2d(2)),
                ('preprocess_res_{}'.format(resolution // 2), prev_preprocess)
            ]))

            new_preprocess = self._get_rgb_block(resolution)
            new_res_block = self._get_intermediate_block(resolution)

            high_res = nn.Sequential()
            high_res.add_module('preprocess_res_{}'.format(resolution), new_preprocess)
            high_res.add_module('res_{}'.format(resolution), new_res_block)

            layers = [
                         ('concat', ConcatLayer(low_res, high_res)),
                         ('fadein', FadeinLayer())
                     ] + res_blocks

            trns_model = nn.Sequential(OrderedDict(layers))
            self.set_net(ProgGrowStageType.trns, resolution, TRNSNetwork(trns_model))

            # stab network
            prev_preprocess = new_preprocess
            res_blocks.insert(0, ('res_{}'.format(resolution), new_res_block))

            layers = [('preprocess_res_{}'.format(resolution), prev_preprocess)] + res_blocks
            stab_model = nn.Sequential(OrderedDict(layers))
            self.set_net(ProgGrowStageType.stab, resolution, STABNetwork(stab_model))

            resolution *= 2

    def _get_last_block_resolution(self):
        if self.output_res == 1:
            return 4
        else:
            return self.output_res

    def _get_num_filters(self, resolution):
        stage = int(math.log(resolution, 2))
        last_stage = int(math.log(self._get_last_block_resolution(), 2))

        return self.inner_dims[-(stage - last_stage) - 1]

    @abstractmethod
    def _get_rgb_block(self, resolution):
        pass

    @abstractmethod
    def _get_last_block(self):
        pass

    @abstractmethod
    def _get_intermediate_block(self, resolution):
        pass


class Resnet9EncoderNetworks(AbstactEncoderNetworks):
    @staticmethod
    def _init_layers(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.2)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return model

    def _get_block(self, block, prev_nf, nf):
        return block(prev_nf, nf, norm=self.norm, act='leaky_relu', pad_type=self.pad_type)

    def _get_intermediate_block(self, resolution):
        prev_nf = self._get_num_filters(resolution)
        nf = self._get_num_filters(resolution // 2)
        layers = [
            self._get_block(PreActResnetBlockDown, prev_nf, nf),
        ]
        return self._init_layers(nn.Sequential(*layers))

    def _get_rgb_block(self, resolution):
        nf = self._get_num_filters(resolution)
        layers = [
            ConvBlock(self.input_dim, nf,
                      kernel_size=3, stride=1, padding=1, norm='none', act='linear', pad_type=self.pad_type),
            self._get_block(PreActResnetBlock, nf, nf),
        ]
        return self._init_layers(nn.Sequential(*layers))

    def _get_last_block(self):
        layers = []
        resolution = self._get_last_block_resolution()
        nf = self._get_num_filters(resolution)

        if self.use_mbstddev:
            layers.append(MinibatchStdDevLayer())

        if self.output_res == 1:
            layers += get_act_layer('leaky_relu')
            layers += [ConvBlock(nf, self.output_dim, kernel_size=4, stride=1, padding=0, norm='none',
                                 pad_type=self.pad_type, act='linear')]
        else:
            layers += [self._get_block(PreActResnetBlock, nf, self.output_dim)]
        return self._init_layers(nn.Sequential(*layers))


class Resnet18EncoderNetworks(Resnet9EncoderNetworks):
    def _get_intermediate_block(self, resolution):
        prev_nf = self._get_num_filters(resolution)
        nf = self._get_num_filters(resolution // 2)
        layers = [
            self._get_block(PreActResnetBlockDown, prev_nf, nf),
            self._get_block(PreActResnetBlock, nf, nf)
        ]
        return self._init_layers(nn.Sequential(*layers))

    def _get_rgb_block(self, resolution):
        nf = self._get_num_filters(resolution)
        layers = [
            ConvBlock(self.input_dim, nf,
                      kernel_size=3, stride=1, padding=1, norm='none', act='linear', pad_type=self.pad_type),
            self._get_block(PreActResnetBlock, nf, nf),
            self._get_block(PreActResnetBlock, nf, nf),
        ]
        return self._init_layers(nn.Sequential(*layers))


class RegularEncoderNetworks(AbstactEncoderNetworks):
    def _get_conv2d(self, in_channels, out_channels, kernel_size, padding):
        return ConvBlock(in_channels, out_channels, kernel_size,
                         stride=1,
                         padding=padding,
                         pad_type=self.pad_type,
                         norm=self.norm,
                         act='leaky_relu',
                         use_wscale=self.use_wscale)

    def _get_rgb_block(self, resolution):
        return ConvBlock(in_channels=self.input_dim,
                         out_channels=self._get_num_filters(min(2 * resolution, self.max_input_res)),
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         act='leaky_relu',
                         use_wscale=self.use_wscale)

    def _get_last_block(self):
        if self.output_res == 1:
            kernel_size, padding = 4, 0
        else:
            kernel_size, padding = 3, 1
        resolution = self._get_last_block_resolution()

        nf = self._get_num_filters(resolution)
        layers = []

        if self.use_mbstddev:
            layers.append(MinibatchStdDevLayer())

        prev_nf = self._get_num_filters(2 * resolution) + self.use_mbstddev

        layers.append(self._get_conv2d(in_channels=prev_nf,
                                       out_channels=nf,
                                       kernel_size=3,
                                       padding=1))
        layers.append(self._get_conv2d(in_channels=nf,
                                       out_channels=nf,
                                       kernel_size=kernel_size,
                                       padding=padding))

        layers.append(ConvBlock(nf, self.output_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                act='linear',
                                use_wscale=self.use_wscale))

        return nn.Sequential(*layers)

    def _get_intermediate_block(self, resolution):
        layers = []
        nf = self._get_num_filters(resolution)
        prev_nf = self._get_num_filters(min(2 * resolution, self.max_input_res))

        layers.append(self._get_conv2d(in_channels=prev_nf,
                                       out_channels=nf,
                                       kernel_size=3,
                                       padding=1))

        layers.append(self._get_conv2d(in_channels=nf,
                                       out_channels=nf,
                                       kernel_size=3,
                                       padding=1))
        layers += get_pool_layer(self.pool, kernel_size=2, stride=2)
        return nn.Sequential(*layers)


ENCODER_NETWORKS = {
    NetworkType.regular: RegularEncoderNetworks,
    NetworkType.residual9: Resnet9EncoderNetworks,
    NetworkType.residual18: Resnet18EncoderNetworks,
}