"""Modules for encoding and decoding observations."""

import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions

from . import util


class ObsEncoder(snt.AbstractModule):
    """Observed -> encoded, flat observation."""

    def __init__(self, hparams, name=None):
        super(ObsEncoder, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams

    @property
    def output_size(self):
        """Returns the output tensor shapes."""
        return tf.TensorShape([self._hparams.enc_obs_size])

    def _build(self, obs):
        hparams = self._hparams
        mlp = util.make_mlp(
            hparams,
            hparams.obs_encoder_fc_layers + [self.output_size.num_elements()])
        return mlp(snt.BatchFlatten()(tf.to_float(obs)))


class ObsDecoder(snt.AbstractModule):
    """inputs -> P(observed | inputs)"""

    def __init__(self, hparams, name=None):
        super(ObsDecoder, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams

    def _build(self, *inputs):
        hparams = self._hparams
        mlp = util.make_mlp(
            hparams,
            hparams.obs_decoder_fc_layers + [np.product(hparams.obs_shape)])
        logits = tf.reshape(
            mlp(util.concat_features(inputs)),
            [-1] + hparams.obs_shape)
        return logits

    @staticmethod
    def output_dist(logits, name=None):
        """Constructs a Distribution from the output of the module."""
        return distributions.OneHotCategorical(
            logits=logits, dtype=tf.int32, name=name)

    def dist(self, *inputs):
        """Returns p(obs | inputs)."""
        return self.output_dist(self(*inputs), name=self.module_name + "Dist")

    @property
    def event_dtype(self):
        """The data type of the observations."""
        return tf.int32
