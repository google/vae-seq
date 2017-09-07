# Copyright 2018 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


class FlattenEncoder(snt.AbstractModule):
    """Forwards the flattened input."""

    def __init__(self, input_size=None, name=None):
        super(FlattenEncoder, self).__init__(name=name)
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


    def _build(self, inp):
        input_sizes = snt.nest.map(lambda inp_i: inp_i.get_shape()[1:], inp)
        self._merge_input_sizes(input_sizes)
        flatten = snt.BatchFlatten(preserve_dims=1)
        flat_inp = snt.nest.map(lambda inp_i: tf.to_float(flatten(inp_i)), inp)
        ret = util.concat_features(flat_inp)
        util.set_tensor_shapes(ret, self.output_size, add_batch_dims=1)
        return ret


def MLPObsEncoder(hparams, name=None):
    """Observation -> encoded, flat observation."""
    name = name or "mlp_obs_encoder"
    mlp = util.make_mlp(hparams, hparams.obs_encoder_fc_layers,
                        name=name + "/mlp")
    return EncoderSequence([FlattenEncoder(), mlp], name=name)


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


class BetaDecoder(dist_module.DistModule):
    """Inputs -> Beta(obs; conc1, conc0)."""

    def __init__(self,
                 positive_projection=None,
                 squeeze_input=False,
                 name=None):
        self._positive_projection = positive_projection
        self._squeeze_input = squeeze_input
        super(BetaDecoder, self).__init__(name=name)

    @property
    def event_dtype(self):
        return tf.float32

    @property
    def event_size(self):
        return tf.TensorShape([])

    def _build(self, inputs):
        conc1, conc0 = tf.split(inputs, 2, axis=-1)
        if self._positive_projection is not None:
            conc1 = self._positive_projection(conc1)
            conc0 = self._positive_projection(conc0)
        if self._squeeze_input:
            conc1 = tf.squeeze(conc1, axis=-1)
            conc0 = tf.squeeze(conc0, axis=-1)
        return (conc1, conc0)

    def dist(self, params, name=None):
        conc1, conc0 = params
        return tf.distributions.Beta(
            conc1, conc0,
            name=name or self.module_name + "_dist")


class _BinomialDist(tf.contrib.distributions.Binomial):
    """Work around missing functionality in Binomial."""

    def __init__(self, total_count, logits=None, probs=None, name=None):
        self._total_count = total_count
        super(_BinomialDist, self).__init__(
            total_count=tf.to_float(total_count),
            logits=logits, probs=probs,
            name=name or "Binomial")

    def _log_prob(self, counts):
        return super(_BinomialDist, self)._log_prob(tf.to_float(counts))

    def _sample_n(self, n, seed=None):
        all_counts = tf.to_float(tf.range(self._total_count + 1))
        for batch_dim in range(self.batch_shape.ndims):
            all_counts = tf.expand_dims(all_counts, axis=-1)
        all_cdfs = tf.map_fn(self.cdf, all_counts)
        shape = tf.concat([[n], self.batch_shape_tensor()], 0)
        uniform = tf.random_uniform(shape, seed=seed)
        return tf.foldl(
            lambda acc, cdfs: tf.where(uniform > cdfs, acc + 1, acc),
            all_cdfs,
            initializer=tf.zeros(shape, dtype=tf.int32))


class BinomialDecoder(dist_module.DistModule):
    """Inputs -> Binomial(obs; total_count, logits)."""

    def __init__(self, total_count=None, squeeze_input=False, name=None):
        self._total_count = total_count
        self._squeeze_input = squeeze_input
        super(BinomialDecoder, self).__init__(name=name)

    @property
    def event_dtype(self):
        return tf.int32

    @property
    def event_size(self):
        return tf.TensorShape([])

    def _build(self, inputs):
        if self._squeeze_input:
            inputs = tf.squeeze(inputs, axis=-1)
        return inputs

    def dist(self, params, name=None):
        return _BinomialDist(
            self._total_count,
            logits=params,
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

    def __init__(self, positive_projection=None, name=None):
        self._positive_projection = positive_projection
        super(NormalDecoder, self).__init__(name=name)

    @property
    def event_dtype(self):
        return tf.float32

    @property
    def event_size(self):
        return tf.TensorShape([])

    def _build(self, inputs):
        loc, scale = tf.split(inputs, 2, axis=-1)
        if self._positive_projection is not None:
            scale = self._positive_projection(scale)
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


class GroupDecoder(dist_module.DistModule):
    """Group up decoders to model a set of independent of events."""

    def __init__(self, decoders, name=None):
        self._decoders = decoders
        super(GroupDecoder, self).__init__(name=name)

    @property
    def event_dtype(self):
        return snt.nest.map(lambda dec: dec.event_dtype, self._decoders)

    @property
    def event_size(self):
        return snt.nest.map(lambda dec: dec.event_size, self._decoders)

    def _build(self, inputs):
        return snt.nest.map_up_to(
            self._decoders,
            lambda dec, input_: dec(input_),
            self._decoders, inputs)

    def dist(self, params, name=None):
        with self._enter_variable_scope():
            with tf.name_scope(name or "group"):
                dists = snt.nest.map_up_to(
                    self._decoders,
                    lambda dec, param: dec.dist(param),
                    self._decoders, params)
            return batch_dist.GroupDistribution(dists, name=name)
