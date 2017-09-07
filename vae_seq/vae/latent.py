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

"""Module parameterizing VAE latent variables."""

import sonnet as snt
import tensorflow as tf

from .. import dist_module
from .. import util


class LatentDecoder(dist_module.DistModule):
    """Inputs -> P(latent | inputs)"""

    def __init__(self, hparams, name=None):
        super(LatentDecoder, self).__init__(name=name)
        self._hparams = hparams

    @property
    def event_dtype(self):
        """The data type of the latent variables."""
        return tf.float32

    def dist(self, params, name=None):
        loc, scale_diag = params
        name = name or self.module_name + "_dist"
        return tf.contrib.distributions.MultivariateNormalDiag(
            loc, scale_diag, name=name)

    def _build(self, *inputs):
        hparams = self._hparams
        mlp = util.make_mlp(
            hparams,
            hparams.latent_decoder_fc_layers + [hparams.latent_size * 2])
        dist_params = mlp(util.concat_features(inputs))
        loc = dist_params[:, :hparams.latent_size]
        scale = util.positive_projection(hparams)(
            dist_params[:, hparams.latent_size:])
        return (loc, scale)
