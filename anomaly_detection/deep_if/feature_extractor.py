import torch
import torch.nn.functional as F
from pretrainedmodels.models import inceptionresnetv2


class InceptionResNetV2FeatureExtractor(torch.nn.Module):
    def __init__(self, feature_extraction_type='default'):
        super().__init__()

        if feature_extraction_type not in [
            'default', 'concat',
            'conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'maxpool_3a',
            "conv2d_3b", "conv2d_4a", "maxpool_5a",
            "mixed_5b",
            "mixed_6a",
            "mixed_7a",
            "conv2d_7b"
        ]:
            raise NotImplementedError("Unknown 'feature_extraction_type': {}".format(feature_extraction_type))

        self.feature_extraction_type = feature_extraction_type
        self.model = inceptionresnetv2()

    def forward(self, x):
        conv2d_1a = self.model.conv2d_1a(x)
        conv2d_2a = self.model.conv2d_2a(conv2d_1a)
        conv2d_2b = self.model.conv2d_2b(conv2d_2a)
        maxpool_3a = self.model.maxpool_3a(conv2d_2b)

        conv2d_3b = self.model.conv2d_3b(maxpool_3a)
        conv2d_4a = self.model.conv2d_4a(conv2d_3b)
        maxpool_5a = self.model.maxpool_5a(conv2d_4a)

        mixed_5b = self.model.mixed_5b(maxpool_5a)
        repeat = self.model.repeat(mixed_5b)

        mixed_6a = self.model.mixed_6a(repeat)
        repeat_1 = self.model.repeat_1(mixed_6a)

        mixed_7a = self.model.mixed_7a(repeat_1)
        repeat_2 = self.model.repeat_2(mixed_7a)

        block8 = self.model.block8(repeat_2)
        conv2d_7b = self.model.conv2d_7b(block8)

        if self.feature_extraction_type == 'default':
            feature_layers = [conv2d_7b]
        elif self.feature_extraction_type == 'concat':
            feature_layers = [conv2d_1a, conv2d_2a, conv2d_2b, maxpool_3a,
                        conv2d_3b, conv2d_4a, maxpool_5a,
                        mixed_5b, mixed_6a, mixed_7a, conv2d_7b]
        else:
            layer_dict = {
                'conv2d_1a': conv2d_1a,
                'conv2d_2a': conv2d_2a,
                'conv2d_2b': conv2d_2b,
                'maxpool_3a': maxpool_3a,
                'conv2d_3b': conv2d_3b,
                'conv2d_4a': conv2d_4a,
                'maxpool_5a': maxpool_5a,
                'mixed_5b': mixed_5b,
                'mixed_6a': mixed_6a,
                'mixed_7a': mixed_7a,
                'conv2d_7b': conv2d_7b,
            }
            feature_layers = [layer_dict[self.feature_extraction_type]]

        features = []
        for layer in feature_layers:
            x = F.adaptive_avg_pool2d(layer, (1, 1))
            x = torch.flatten(x, 1)
            features.append(x)
        features = torch.cat(features, dim=1)
        return features
