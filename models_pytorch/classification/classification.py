import torch
from torch import nn as nn

from .classifiers import get_classifier
from ..base import Model
from ..encoders import get_encoder


class ClassificationModel(Model):

    def __init__(self, encoder='resnet34', activation='sigmoid',
                 encoder_weights="imagenet", classes=6,
                 classifiers_params=None, model_dir=None,
                 **kwargs):
        super().__init__()

        classifiers_params = classifiers_params or {'cls0': {'type': 'basic'}}

        if not isinstance(classifiers_params, dict):
            classifiers_params = {'cls{}'.format(i): cp for i, cp in enumerate(classifiers_params)}

        self.encoder = get_encoder(encoder, encoder_weights=encoder_weights, model_dir=model_dir)

        for d in classifiers_params.values():
            d.setdefault('activation', activation)
            d.setdefault('classes', classes)
            d['encoder_channels'] = self.encoder.out_shapes
            d.update(**kwargs)

        self.classifiers = nn.ModuleDict({name: get_classifier(**args) for name, args in classifiers_params.items()})

    def forward(self, x, **args):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        features = self.encoder(x)
        outputs = {name: classifier.forward(features) for name, classifier in self.classifiers.items()}
        return outputs

    def predict(self, x, **args):

        if self.training:
            self.eval()
        with torch.no_grad():
            features = self.encoder(x, **args)
            return {name: classifier.predict(features) for name, classifier in self.classifiers.items()}
