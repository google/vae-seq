"""Modules for encoding and decoding observations."""

import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions

from . import util


class IdentityObsEncoder(snt.AbstractModule):
    """Forwards the (flattened) input observation."""

    def __init__(self, hparams, name=None):
        super(IdentityObsEncoder, self).__init__(name=name)
        self._hparams = hparams

    @property
    def output_size(self):
        """Returns the output Tensor shapes."""
        return tf.TensorShape([np.product(self._hparams.obs_shape)])

    def _build(self, obs):
        return snt.BatchFlatten()(tf.to_float(obs))


class MLPObsEncoder(snt.AbstractModule):
    """Observation -> encoded, flat observation."""

    def __init__(self, hparams, name=None):
        super(MLPObsEncoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._mlp = util.make_mlp(
                hparams,
                hparams.obs_encoder_fc_layers)

    @property
    def output_size(self):
        """Returns the output Tensor shapes."""
        return self._mlp.output_size

    def _build(self, obs):
        return self._mlp(snt.BatchFlatten()(tf.to_float(obs)))


class OneHotObsDecoder(snt.AbstractModule):
    """inputs -> Categorical(observed; logits=mlp(inputs))"""

    def __init__(self, hparams, name=None):
        super(OneHotObsDecoder, self).__init__(name=name)
        self._hparams = hparams

    def _build(self, *inputs):
        hparams = self._hparams
        layers = (hparams.obs_decoder_fc_hidden_layers +
                  [np.product(hparams.obs_shape)])
        mlp = util.make_mlp(hparams, layers)
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
