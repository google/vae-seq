"""Modules for encoding and decoding observations."""

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

    @property
    def output_size(self):
        return tf.TensorShape(self._hparams.obs_shape)

    def _build(self, *inputs):
        hparams = self._hparams
        mlp = util.make_mlp(
            hparams,
            hparams.obs_decoder_fc_layers + [self.output_size.num_elements()])
        logits = tf.reshape(
            mlp(util.concat_features(inputs)),
            [-1] + self.output_size.as_list())
        return logits

    @staticmethod
    def output_dist(logits, name=None):
        return distributions.OneHotCategorical(
            logits=logits, dtype=tf.int32, name=name)

    def dist(self, *inputs):
        return self.output_dist(self(*inputs), name=self.module_name + "Dist")

    @property
    def event_dtype(self):
        return tf.int32
