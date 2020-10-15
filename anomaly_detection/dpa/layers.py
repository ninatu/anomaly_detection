import numpy as np
import torch
from torch import nn as nn


def get_norm_layer(type, **kwargs):
    if type == 'none':
        return []
    elif type == 'bn':
        return [nn.BatchNorm2d(kwargs['num_features'])]
    elif type == 'in':
        return [nn.InstanceNorm2d(kwargs['num_features'])]
    elif type == 'pixel':
        return [PixelNormLayer()]
    else:
        raise NotImplementedError("Unknown type: {}".format(type))


def get_act_layer(type, **kwargs):
    if type == 'relu':
        return [nn.ReLU()]
    elif type == 'leaky_relu':
        return [nn.LeakyReLU(kwargs.get('negative_slope', 0.2), inplace=False)]
    elif type == 'tanh':
        return [nn.Tanh()]
    elif type == 'sigmoid':
        return [nn.Sigmoid()]
    elif type == 'linear':
        return []
    else:
        raise NotImplementedError("Unknown type: {}".format(type))


def get_pool_layer(type, **kwargs):
    if type == 'avg':
        return [nn.AvgPool2d(kwargs.get('kernel_size', 2), kwargs.get('stride', 2))]
    elif type == 'max':
        return [nn.MaxPool2d(kwargs.get('kernel_size', 2), kwargs.get('stride', 2))]
    else:
        raise NotImplementedError("Unknown type: {}".format(type))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pad_type='zero', norm='none',
                 act='linear', use_wscale=False):
        super(ConvBlock, self).__init__()
        leaky_relu_param = 0.2
        layers = []

        if pad_type == 'reflect':
            layers.append(nn.ReflectionPad2d(padding))
            padding = 0
        elif pad_type == 'zero':
            pass
        else:
            raise NotImplementedError

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain(act, param=leaky_relu_param))
        layers.append(conv)

        if use_wscale:
            layers.append(WScaleLayer(conv))

        layers += get_norm_layer(norm, num_features=out_channels)
        layers += get_act_layer(act, negative_slope=leaky_relu_param)

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pad_type='zero', lambd=1):
        super(ResBlock, self).__init__()
        self.lambd = lambd

        model = []
        model += [ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act, pad_type=pad_type)]
        model += [ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear', pad_type=pad_type)]

        self.model = nn.Sequential(*model)

        if input_dim == output_dim:
            self.skipcon = nn.Sequential()
        else:
            self.skipcon = ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear', pad_type=pad_type)

    def forward(self, x):
        return self.skipcon(x) + self.lambd * self.model(x)


class PreActResnetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pad_type='zero', lambd=1):
        super(PreActResnetBlock, self).__init__()
        self.lambd = lambd

        model = []
        model += get_norm_layer(norm, num_features=input_dim)
        model += get_act_layer(act)
        model += [
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act, pad_type=pad_type),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear', pad_type=pad_type)
        ]

        self.model = nn.Sequential(*model)

        if input_dim == output_dim:
            self.skipcon = nn.Sequential()
        else:
            self.skipcon = ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear', pad_type=pad_type)

    def forward(self, x):
        return self.skipcon(x) + self.lambd * self.model(x)


class PreActResnetBlockUp(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pad_type='zero', upsample_mode='nearest'):
        super(PreActResnetBlockUp, self).__init__()

        model = []
        model += get_norm_layer(norm, num_features=input_dim)
        model += get_act_layer(act)
        model += [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act, pad_type=pad_type),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear', pad_type=pad_type)
        ]

        self.model = nn.Sequential(*model)

        skipcon = [nn.Upsample(scale_factor=2, mode='nearest')]
        if input_dim != output_dim:
            skipcon += [ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear', pad_type=pad_type)]
        self.skipcon = nn.Sequential(*skipcon)

    def forward(self, x):
        return self.skipcon(x) + self.model(x)


class PreActResnetBlockDown(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pad_type='zero', pool='avg'):
        super(PreActResnetBlockDown, self).__init__()

        model = []
        model += get_norm_layer(norm, num_features=input_dim)
        model += get_act_layer(act)
        model += get_pool_layer(pool, kernel_size=2, stride=2)
        model += [
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act, pad_type=pad_type),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear', pad_type=pad_type),
        ]
        self.model = nn.Sequential(*model)

        skipcon = get_pool_layer(pool, kernel_size=2, stride=2)
        if input_dim != output_dim:
            skipcon += [ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear', pad_type=pad_type)]
        self.skipcon = nn.Sequential(*skipcon)

    def forward(self, x):
        return self.skipcon(x) + self.model(x)


# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans
class WScaleLayer(nn.Module):
    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming

        scale = (torch.mean(self.incoming.weight.data ** 2)) ** 0.5
        self.scale = nn.Parameter(torch.Tensor([scale]), requires_grad=False)
        self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale.data)

        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = self.scale * x
        if self.bias is not None:
            if isinstance(self.incoming, nn.Linear):
                x += self.bias.view(1, self.bias.size()[0])
            elif isinstance(self.incoming, nn.Conv2d):
                x += self.bias.view(1, self.bias.size()[0], 1, 1)
            else:
                raise NotImplementedError
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % (self.incoming.__class__.__name__)
        return self.__class__.__name__ + param_str


# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans
class MinibatchStdDevLayer(nn.Module):
    def __init__(self, save_vals=False):
        super(MinibatchStdDevLayer, self).__init__()
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)
        self.save_vals = save_vals
        self.saved_vals = None

    def forward(self, x):
        target_shape = list(x.size())
        target_shape[1] = 1

        vals = self.adjusted_std(x, dim=0, keepdim=True)
        vals = torch.mean(vals)

        if self.save_vals:
            self.saved_vals = vals.data

        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)


# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans
class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """

    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


class FadeinLayer(nn.Module):
    def __init__(self, ):
        super(FadeinLayer, self).__init__()
        self._alpha = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=False)

    def set_progress(self, alpha):
        self._alpha.data[0] = np.clip(alpha, 0, 1.0)

    def get_progress(self):
        return self._alpha.data.cpu().item()

    def forward(self, x):
        return torch.add(x[0].mul(1.0 - self._alpha), x[1].mul(self._alpha))

    def __repr__(self):
        return self.__class__.__name__ + '(get_alpha = {:.2f})'.format(self._alpha.data[0])


class ConcatLayer(nn.Module):
    def __init__(self, layer1, layer2):
        super(ConcatLayer, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        y = [self.layer1(x), self.layer2(x)]
        return y


class EqualLayer(nn.Module):
    def forward(self, x):
        return x
