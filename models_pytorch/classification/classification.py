import six
import torch
from torch import nn as nn

from .classifiers import get_classifier
from ..base import Model
from ..encoders import get_encoder
from ..utils import get_activation


class ClassificationModel(Model):

    def __init__(self, encoder='resnet34', activation='sigmoid',
                 encoder_weights="imagenet", classes=6,
                 classifiers_params=None, model_dir=None,

                 **kwargs):
        super().__init__()

        classifiers_params = classifiers_params or [{'type': 'basic'}, ]
        if isinstance(activation, six.string_types):
            activation = [activation, ] * len(classifiers_params)

        if isinstance(classes, six.integer_types):
            classes = [classes, ] * len(classifiers_params)

        assert len(activation) == len(classifiers_params)

        self.encoder = get_encoder(encoder, encoder_weights=encoder_weights, model_dir=model_dir)
        self.classifiers = nn.ModuleList([get_classifier(encoder_channels=self.encoder.out_shapes,
                                                         classes=cl, **cp) for cp, cl in
                                          zip(classifiers_params, classes)])
        self.activations = [get_activation(a) for a in activation]

    def forward(self, x, **args):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        features = self.encoder(x)
        outputs = [classifier(features) for classifier in self.classifiers]
        return outputs

    def predict(self, x, **args):

        if self.training:
            self.eval()
        with torch.no_grad():
            outputs = self.forward(x, **args)
            scores = [act(output) for output, act in zip(outputs, self.activations)]
            return scores
