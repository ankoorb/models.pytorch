import torch
import torch.nn as nn

from models_pytorch.utils import Flatten, get_activation


class BasicClassifier(nn.Module):

    pooling_types = {
        'avg': nn.AdaptiveAvgPool2d,
        'max': nn.AdaptiveMaxPool2d,
    }

    def __init__(self, encoder_channels, classes, hidden_layers=(), pool_type='avg', channel_index=0,
                 activation='sigmoid'):
        hidden_layers = hidden_layers or []

        self.channel_index = channel_index
        input_shape = encoder_channels[self.channel_index]

        input_shapes = [input_shape]
        input_shapes.extend(hidden_layers)

        output_shapes = list(hidden_layers)
        output_shapes.append(classes)

        super().__init__()
        modules = [self.pooling_types[pool_type]((1, 1)), Flatten()]
        for ih, oh in zip(input_shapes, output_shapes):
            modules.extend([nn.Linear(ih, oh),
                            nn.ReLU()])

        modules.pop(-1)
        self.classifier = nn.Sequential(*modules)
        self.activation = get_activation(activation)

    def forward(self, features):
        return self.classifier(features[self.channel_index])

    def predict(self, features):
        if self.training:
            self.eval()
        with torch.no_grad():
            return self.activation(self.forward(features))


classifier_types = {'basic': BasicClassifier}


def get_classifier(type, **kwargs):
    return classifier_types[type](**kwargs)


