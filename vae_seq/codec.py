"""Modules for encoding and decoding observations."""

import abc
import numpy as np
import sonnet as snt
import tensorflow as tf

from . import dist_module
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


class OneHotObsDecoder(dist_module.DistModule):
    """Inputs -> Categorical(observed; logits=mlp(inputs))"""

    def __init__(self, hparams, name=None):
        super(OneHotObsDecoder, self).__init__(name=name)
        self._hparams = hparams

    @property
    def event_dtype(self):
        return tf.int32

    def dist(self, logits, name=None):
        """Constructs a Distribution from the output of the module."""
        name = name or self.module_name + "_dist"
        return tf.contrib.distributions.OneHotCategorical(
            logits=logits, dtype=tf.int32, name=name)

    def _build(self, *inputs):
        hparams = self._hparams
        layers = (hparams.obs_decoder_fc_hidden_layers +
                  [np.product(hparams.obs_shape)])
        mlp = util.make_mlp(hparams, layers)
        logits = tf.reshape(
            mlp(util.concat_features(inputs)),
            [-1] + hparams.obs_shape)
        logits.set_shape([None] + hparams.obs_shape)
        return logits
