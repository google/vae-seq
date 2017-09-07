# Copyright 2017 Google, Inc.,
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

"""Coders/Decoders for Audio samples."""

import functools
import numpy as np
import tensorflow as tf
import sonnet as snt
from vae_seq import dist_module
from vae_seq import util


class AudioObsEncoder(snt.AbstractModule):
    """1D Convolutions followed by an MLP."""

    def __init__(self, hparams, name=None):
        super(AudioObsEncoder, self).__init__(name=name)
        self._hparams = hparams
        with self._enter_variable_scope():
            self._mlp = util.make_mlp(
                hparams,
                hparams.obs_encoder_fc_layers)

    @property
    def output_size(self):
        """Returns the output Tensor shapes."""
        return self._mlp.output_size

    def _build(self, samples):
        hparams = self._hparams
        layers = [functools.partial(tf.expand_dims, axis=2)]
        conv_params = np.array(hparams.obs_enc_conv_layers).reshape((-1, 4))
        for conv_layer_params in conv_params:
            (channels, kernel, pool_size, strides) = conv_layer_params
            layers += [snt.Conv1D(channels, kernel_shape=kernel),
                       functools.partial(tf.layers.max_pooling1d,
                                         pool_size=pool_size,
                                         strides=strides)]
        layers += [snt.BatchFlatten(), self._mlp]
        return snt.Sequential(layers)(samples)


class AudioObsDecoder(dist_module.DistModule):
    """Inputs -> Diagonal MVN(observed; mu, var).

    The parameters of mu and var are determined by a decoder MLP,
    followed by transposed convolutions layers.
    """

    def __init__(self, hparams, name=None):
        super(AudioObsDecoder, self).__init__(name=name)
        self._hparams = hparams

    @property
    def event_dtype(self):
        """The data type of the observations."""
        return tf.float32

    def dist(self, params, name=None):
        """Constructs a Distribution from the output of the module."""
        loc, scale = params
        name = name or self.module_name + "_dist"
        return tf.contrib.distributions.MultivariateNormalDiag(
            loc, scale, name=name)

    def _build(self, *inputs):
        hparams = self._hparams
        deconv_params = np.array(hparams.obs_dec_deconv_layers).reshape((-1, 3))
        # Calculate the required widths and output channels to end up with
        # [-1, obs_shape, 2]
        reversed_width_and_channels = [(hparams.obs_shape[0], 2)]
        for deconv_layer_params in deconv_params[::-1]:
            input_channels, _unused_kernel_shape, stride = deconv_layer_params
            output_width = reversed_width_and_channels[-1][0]
            input_width = (output_width + stride - 1) // stride
            reversed_width_and_channels.append((input_width, input_channels))
        width_and_channels = list(reversed(reversed_width_and_channels))
        mlp = util.make_mlp(
            hparams,
            hparams.obs_decoder_fc_hidden_layers + [
                width_and_channels[0][0] * width_and_channels[0][1]])
        layers = [
            util.concat_features,
            mlp,
            snt.BatchReshape(width_and_channels[0]),
        ]
        for layer_params in zip(width_and_channels[1:], deconv_params):
            output_width, channels = layer_params[0]
            _unused_input_channels, kernel_shape, stride = layer_params[1]
            deconv = snt.Conv1DTranspose(channels,
                                         output_shape=output_width,
                                         kernel_shape=kernel_shape,
                                         stride=stride)
            layers.append(deconv)
        params = snt.Sequential(layers)(inputs)
        loc, unproj_scale = tf.unstack(params, axis=2)
        return loc, util.positive_projection(hparams)(unproj_scale)
