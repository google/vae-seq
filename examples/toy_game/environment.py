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

"""Environment that runs the test game."""

import pickle
import numpy as np
import tensorflow as tf
from vae_seq import agent as agent_mod

from . import game as game_mod


class Environment(agent_mod.Environment):
    """Plays a batch of games, keeping pickled game states as RNN state."""

    def __init__(self, hparams, name=None):
        super(Environment, self).__init__(name=name)
        self._hparams = hparams

    @property
    def output_size(self):
        return tf.TensorShape(self._hparams.obs_shape)

    @property
    def output_dtype(self):
        return tf.int32

    @property
    def state_size(self):
        return tf.TensorShape([])  # pickled games.

    def initial_state(self, batch_size):
        hparams = self._hparams

        def _random_games(batch_size):
            """Produces a serialized batch of randomized games."""
            return np.array(
                [
                    pickle.dumps(
                        game_mod.Game(hparams.toy_game_width,
                                      hparams.toy_game_classes))
                    for _ in range(batch_size)
                ],
                dtype=object)

        state, = tf.py_func(_random_games, [batch_size], [tf.string])
        state.set_shape([None])
        return state

    def _build(self, actions, state):

        def _step(actions, state):
            """Applies the batch actions to the batch of serialized games."""
            games = [pickle.loads(game) for game in state]
            for action, game in zip(actions, games):
                game.take_action(action)
            obs = [game.render() for game in games]
            state = [pickle.dumps(game) for game in games]
            return np.array(obs, dtype=np.int32), np.array(state, dtype=object)

        out, state = tf.py_func(_step, [actions, state], [tf.int32, tf.string])
        out.set_shape([None] + self._hparams.obs_shape)
        state.set_shape([None])
        return out, state
