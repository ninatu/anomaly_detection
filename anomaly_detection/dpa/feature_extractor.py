import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from anomaly_detection.dpa.layers import EqualLayer
from torchvision.models import vgg19


class PretrainedVGG19FeatureExtractor(nn.Module):
    def __init__(self, pad_type='zero'):
        super(PretrainedVGG19FeatureExtractor, self).__init__()
        self.pad_type = pad_type

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(1)
            padding = 0
        elif pad_type == 'zero':
            self.pad = EqualLayer()
            padding = 1
        elif pad_type == 'replication':
            self.pad = nn.ReplicationPad2d(1)
            padding = 0
        else:
            raise NotImplementedError

        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=padding)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=padding)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=padding)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=padding)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=padding)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=padding)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=padding)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=padding)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=padding)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        for param in self.parameters():
            param.requires_grad = False

        # loading weights
        pretrained_features = vgg19(pretrained=True).features
        assert len(pretrained_features.state_dict().keys()) == len(self.state_dict().keys())

        state_dict = OrderedDict()
        for (new_name, _), (_, value) in zip(self.state_dict().items(), pretrained_features.state_dict().items()):
            state_dict[new_name] = value

        self.load_state_dict(state_dict)

    def forward(self, x, out_keys):
        out = {}
        
        def finished():
            return len(set(out_keys).difference(out.keys())) == 0

        out['c11'] = self.conv1_1(self.pad(x)) if not(finished()) else None
        out['r11'] = F.relu(out['c11']) if not(finished()) else None
        out['c12'] = self.conv1_2(self.pad(out['r11'])) if not(finished()) else None
        out['r12'] = F.relu(out['c12']) if not(finished()) else None
        out['p1'] = self.pool1(out['r12']) if not(finished()) else None

        out['c21'] = self.conv2_1(self.pad(out['p1'])) if not(finished()) else None
        out['r21'] = F.relu(out['c21']) if not(finished()) else None
        out['c22'] = self.conv2_2(self.pad(out['r21'])) if not(finished()) else None
        out['r22'] = F.relu(out['c22']) if not(finished()) else None
        out['p2'] = self.pool2(out['r22']) if not(finished()) else None

        out['c31'] = self.conv3_1(self.pad(out['p2'])) if not(finished()) else None
        out['r31'] = F.relu(out['c31']) if not(finished()) else None
        out['c32'] = self.conv3_2(self.pad(out['r31'])) if not(finished()) else None
        out['r32'] = F.relu(out['c32']) if not(finished()) else None
        out['c33'] = self.conv3_3(self.pad(out['r32'])) if not(finished()) else None
        out['r33'] = F.relu(out['c33']) if not(finished()) else None
        out['c34'] = self.conv3_4(self.pad(out['r33'])) if not(finished()) else None
        out['r34'] = F.relu(out['c34']) if not(finished()) else None
        out['p3'] = self.pool3(out['r34']) if not(finished()) else None

        out['c41'] = self.conv4_1(self.pad(out['p3'])) if not(finished()) else None
        out['r41'] = F.relu(out['c41']) if not(finished()) else None
        out['c42'] = self.conv4_2(self.pad(out['r41'])) if not(finished()) else None
        out['r42'] = F.relu(out['c42']) if not(finished()) else None
        out['c43'] = self.conv4_3(self.pad(out['r42'])) if not(finished()) else None
        out['r43'] = F.relu(out['c43']) if not(finished()) else None
        out['c44'] = self.conv4_4(self.pad(out['r43'])) if not(finished()) else None
        out['r44'] = F.relu(out['c44']) if not(finished()) else None
        out['p4'] = self.pool4(out['r44']) if not(finished()) else None

        out['c51'] = self.conv5_1(self.pad(out['p4'])) if not(finished()) else None
        out['r51'] = F.relu(out['c51']) if not(finished()) else None
        out['c52'] = self.conv5_2(self.pad(out['r51'])) if not(finished()) else None
        out['r52'] = F.relu(out['c52']) if not(finished()) else None
        out['c53'] = self.conv5_3(self.pad(out['r52'])) if not(finished()) else None
        out['r53'] = F.relu(out['c53']) if not(finished()) else None
        out['c54'] = self.conv5_4(self.pad(out['r53'])) if not(finished()) else None
        out['r54'] = F.relu(out['c54']) if not(finished()) else None
        out['p5'] = self.pool5(out['r54']) if not(finished()) else None
        return [out[key] for key in out_keys]
