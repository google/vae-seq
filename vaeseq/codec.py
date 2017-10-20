"""Modules for encoding and decoding observations."""

import sonnet as snt
import tensorflow as tf

from . import batch_dist
from . import dist_module
from . import util


class EncoderSequence(snt.Sequential):
    """A wrapper arount snt.Sequential that also implements output_size."""

    @property
    def output_size(self):
        return self.layers[-1].output_size


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
        util.set_tensor_shapes(ret, self.output_size, add_batch_dims=1)
        return ret


def MLPObsEncoder(hparams, name=None):
    """Observation -> encoded, flat observation."""
    name = name or "mlp_obs_encoder"
    mlp = util.make_mlp(hparams, hparams.obs_encoder_fc_layers,
                        name=name + "/mlp")
    return EncoderSequence([FlattenObsEncoder(), mlp], name=name)


class DecoderSequence(dist_module.DistModule):
    """A sequence of zero or more AbstractModules, followed by a DistModule."""

    def __init__(self, input_encoders, decoder, name=None):
        super(DecoderSequence, self).__init__(name=name)
        self._input_encoders = input_encoders
        self._decoder = decoder

    @property
    def event_dtype(self):
        return self._decoder.event_dtype

    @property
    def event_size(self):
        return self._decoder.event_size

    def dist(self, params, name=None):
        return self._decoder.dist(params, name=name)

    def _build(self, inputs):
        if self._input_encoders:
            inputs = snt.Sequential(self._input_encoders)(inputs)
        return self._decoder(inputs)


def MLPObsDecoder(hparams, decoder, param_size, name=None):
    """Inputs -> decoder(obs; mlp(inputs))."""
    name = name or "mlp_" + decoder.module_name
    layers = hparams.obs_decoder_fc_hidden_layers + [param_size]
    mlp = util.make_mlp(hparams, layers, name=name + "/mlp")
    return DecoderSequence([util.concat_features, mlp], decoder, name=name)


class BernoulliDecoder(dist_module.DistModule):
    """Inputs -> Bernoulli(obs; logits=inputs)."""

    def __init__(self, dtype=tf.int32, squeeze_input=False, name=None):
        self._dtype = dtype
        self._squeeze_input = squeeze_input
        super(BernoulliDecoder, self).__init__(name=name)

    @property
    def event_dtype(self):
        return self._dtype

    @property
    def event_size(self):
        return tf.TensorShape([])

    def _build(self, inputs):
        if self._squeeze_input:
            inputs = tf.squeeze(inputs, axis=-1)
        return inputs

    def dist(self, params, name=None):
        return tf.distributions.Bernoulli(
            logits=params,
            dtype=self._dtype,
            name=name or self.module_name + "_dist")


class CategoricalDecoder(dist_module.DistModule):
    """Inputs -> Categorical(obs; logits=inputs)."""

    def __init__(self, dtype=tf.int32, name=None):
        self._dtype = dtype
        super(CategoricalDecoder, self).__init__(name=name)

    @property
    def event_dtype(self):
        return self._dtype

    @property
    def event_size(self):
        return tf.TensorShape([])

    def _build(self, inputs):
        return inputs

    def dist(self, params, name=None):
        return tf.distributions.Categorical(
            logits=params,
            dtype=self._dtype,
            name=name or self.module_name + "_dist")


class NormalDecoder(dist_module.DistModule):
    """Inputs -> Normal(obs; loc=half(inputs), scale=project(half(inputs)))"""

    def __init__(self, hparams, name=None):
        self._hparams = hparams
        super(NormalDecoder, self).__init__(name=name)

    @property
    def event_dtype(self):
        return tf.float32

    @property
    def event_size(self):
        return tf.TensorShape([])

    def _build(self, inputs):
        loc, unproj_scale = tf.split(inputs, 2, axis=-1)
        scale = util.positive_projection(self._hparams)(unproj_scale)
        return loc, scale

    def dist(self, params, name=None):
        loc, scale = params
        return tf.distributions.Normal(
            loc=loc,
            scale=scale,
            name=name or self.module_name + "_dist")


class BatchDecoder(dist_module.DistModule):
    """Wrap a decoder to model batches of events."""

    def __init__(self, decoder, event_size, name=None):
        self._decoder = decoder
        self._event_size = tf.TensorShape(event_size)
        super(BatchDecoder, self).__init__(name=name)

    @property
    def event_dtype(self):
        return self._decoder.event_dtype

    @property
    def event_size(self):
        return self._event_size

    def _build(self, inputs):
        return self._decoder(inputs)

    def dist(self, params, name=None):
        return batch_dist.BatchDistribution(
            self._decoder.dist(params, name=name),
            ndims=self._event_size.ndims)
