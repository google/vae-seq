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

"""Coders/Decoders for game observations."""

import numpy as np
import sonnet as snt
import tensorflow as tf

from vaeseq import batch_dist
from vaeseq import codec
from vaeseq import dist_module
from vaeseq import util


ObsEncoder = codec.MLPObsEncoder


class InputEncoder(codec.FlattenEncoder):
    """Passes through the input action logits."""

    def __init__(self, hparams, name=None):
        input_size = tf.TensorShape([hparams.game_action_space])
        super(InputEncoder, self).__init__(input_size=input_size, name=name)


class ObsDecoder(dist_module.DistModule):
    """Parameterizes a set of distributions for outputs, score and game-over.

    We're modeling three components for each observation:

    * The game outputs modeled by a diagonal multivariate normal.
    * The current score (a Normal distribution).
    * Whether the game is over next step. For simplicity, modeled as a normal
      with -1/+1 labels.

    All output distributions are reparameterizable, so there is a
    pathwise derivative w.r.t. their parameters.
    """

    def __init__(self, hparams, name=None):
        super(ObsDecoder, self).__init__(name=name)
        self._hparams = hparams

    @property
    def event_size(self):
        return dict(output=tf.TensorShape(self._hparams.game_output_size),
                    score=tf.TensorShape([]),
                    game_over=tf.TensorShape([]))

    @property
    def event_dtype(self):
        return dict(output=tf.float32,
                    score=tf.float32,
                    game_over=tf.float32)

    def dist(self, params, name=None):
        """The output distribution."""
        name = name or self.module_name + "_dist"
        with tf.name_scope(name):
            params_output, params_score, params_game_over = params
            components = dict(
                output=self._dist_output(params_output),
                score=self._dist_score(params_score),
                game_over=self._dist_game_over(params_game_over))
        return batch_dist.GroupDistribution(components, name=name)

    def _dist_output(self, params):
        """Distribution over the game outputs."""
        loc, scale_diag = params
        return tf.contrib.distributions.MultivariateNormalDiag(
            loc, scale_diag, name="game_output")

    def _dist_score(self, params):
        """Distribution for the game score."""
        loc, scale = params
        return tf.distributions.Normal(loc, scale, name="score")

    def _dist_game_over(self, params):
        """Distribution for the game over flag."""
        loc, scale = params
        return tf.distributions.Normal(loc, scale, name="game_over")

    def _build(self, inputs):
        hparams = self._hparams
        hidden = snt.Sequential([
            util.concat_features,
            util.make_mlp(
                hparams,
                hparams.obs_decoder_fc_hidden_layers,
                activate_final=True),
        ])(inputs)
        return (self._build_game_output(hidden),
                self._build_score(hidden),
                self._build_game_over(hidden))

    def _build_game_output(self, hidden):
        """Parameters for the game output prediction."""
        game_outputs = np.product(self._hparams.game_output_size)
        lin = snt.Linear(2 * game_outputs, name="game_obs")
        loc, scale_diag_unproj = tf.split(lin(hidden), 2, axis=-1)
        scale_diag = util.positive_projection(self._hparams)(scale_diag_unproj)
        return loc, scale_diag

    def _build_score(self, hidden):
        """Parameters for the game score prediction."""
        lin = snt.Linear(2, name="score")
        loc, scale_unproj = tf.unstack(lin(hidden), axis=-1)
        scale = util.positive_projection(self._hparams)(scale_unproj)
        return loc, scale

    def _build_game_over(self, hidden):
        """Parameters for the game over prediction."""
        lin = snt.Linear(2, name="game_over")
        loc, scale_unproj = tf.unstack(lin(hidden), axis=-1)
        scale = util.positive_projection(self._hparams)(scale_unproj)
        return loc, scale
