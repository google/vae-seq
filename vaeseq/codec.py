"""Modules for encoding and decoding observations."""

import abc
import numpy as np
import sonnet as snt
import tensorflow as tf

from . import dist_module
from . import util


class FlattenObsEncoder(snt.AbstractModule):
    """Forwards the (flattened) input observation."""

    def __init__(self, input_size=None, name=None):
        super(FlattenObsEncoder, self).__init__(name=name)
        self._input_size = None
        if input_size is not None:
            self._merge_input_sizes(input_size)

    def _merge_input_sizes(self, input_size):
        if self._input_size is None:
            self._input_size = snt.nest.map(tf.TensorShape, input_size)
            return
        self._input_size = snt.nest.map(
            lambda cur_size, inp_size: cur_size.merge_with(inp_size),
            self._input_size,
            input_size)

    @property
    def output_size(self):
        """Returns the output Tensor shapes."""
        if self._input_size is None:
            return tf.TensorShape([None])
        flattened_size = 0
        for inp_size in snt.nest.flatten(self._input_size):
            num_elements = inp_size.num_elements()
            if num_elements is None:
                return tf.TensorShape([None])
            flattened_size += num_elements
        return tf.TensorShape([flattened_size])


    def _build(self, obs):
        input_sizes = snt.nest.map(lambda obs_i: obs_i.get_shape()[1:], obs)
        self._merge_input_sizes(input_sizes)
        flatten = snt.BatchFlatten(preserve_dims=1)
        flat_obs = snt.nest.map(lambda obs_i: tf.to_float(flatten(obs_i)), obs)
        ret = util.concat_features(flat_obs)
        ret.set_shape(tf.TensorShape([None]).concatenate(self.output_size))
        return ret


class MLPObsEncoder(snt.AbstractModule):
    """Observation -> encoded, flat observation."""

    def __init__(self, hparams, name=None):
        super(MLPObsEncoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._flatten = FlattenObsEncoder()
            self._mlp = util.make_mlp(
                hparams,
                hparams.obs_encoder_fc_layers)

    @property
    def output_size(self):
        """Returns the output Tensor shapes."""
        return self._mlp.output_size

    def _build(self, obs):
        return self._mlp(self._flatten(obs))


class MLPObsDecoderBase(dist_module.DistModule):
    """Base class for MLP -> Distribution decoders."""

    def __init__(self, hparams, param_size, name=None):
        super(MLPObsDecoderBase, self).__init__(name=name)
        self._hparams = hparams
        self._param_size = param_size

    def _build(self, *inputs):
        hparams = self._hparams
        layers = hparams.obs_decoder_fc_hidden_layers + [self._param_size]
        mlp = util.make_mlp(hparams, layers)
        return mlp(util.concat_features(inputs))


class BernoulliMLPObsDecoder(MLPObsDecoderBase):
    """Inputs -> Bernoulli(obs; logits=mlp(inputs))."""

    def __init__(self, hparams, dtype=tf.int32, name=None):
        self._dtype = dtype
        super(BernoulliMLPObsDecoder, self).__init__(hparams, 1, name=name)

    @property
    def event_dtype(self):
        return self._dtype

    @property
    def event_size(self):
        return tf.TensorShape([])

    def dist(self, params, name=None):
        return tf.distributions.Bernoulli(
            logits=tf.squeeze(params, axis=-1),
            dtype=self._dtype,
            name=name or self.module_name + "_dist")


class CategoricalMLPObsDecoder(MLPObsDecoderBase):
    """Inputs -> Categorical(obs; logits=mlp(inputs))."""

    def __init__(self, hparams, num_classes, dtype=tf.int32, name=None):
        self._dtype = dtype
        super(CategoricalMLPObsDecoder, self).__init__(
            hparams, num_classes, name=name)

    @property
    def event_dtype(self):
        return self._dtype

    @property
    def event_size(self):
        return tf.TensorShape([])

    def dist(self, params, name=None):
        return tf.distributions.Categorical(
            logits=params,
            dtype=self._dtype,
            name=name or self.module_name + "_dist")
