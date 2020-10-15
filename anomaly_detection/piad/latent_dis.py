from torch import nn

from anomaly_detection.dpa.layers import ConvBlock


class LatentDiscriminator(nn.Module):
    def __init__(self, input_dim, inner_dims, output_dim=1, norm='none', act='leaky_relu'):
        super(LatentDiscriminator, self).__init__()

        layers = list()
        inner_dims = [input_dim] + inner_dims
        dim = inner_dims[0]

        for next_dim in inner_dims[1:]:
            layers.append(ConvBlock(dim, next_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     norm=norm,
                                     act=act))
            dim = next_dim

        layers.append(ConvBlock(dim, output_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm='none',
                                act='linear'))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1, 1, 1)
        return self.model(x)
