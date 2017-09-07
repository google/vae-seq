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

"""Game-playing agent."""

import abc
import sonnet as snt
import tensorflow as tf

from vaeseq import context as context_mod
from vaeseq import util


class AgentBase(context_mod.Context):
    """Base class for input agents."""

    def __init__(self, hparams, name=None):
        super(AgentBase, self).__init__(name=name)
        self._hparams = hparams
        self._num_actions = tf.TensorShape([self._hparams.game_action_space])

    @property
    def output_size(self):
        return self._num_actions

    @property
    def output_dtype(self):
        return tf.float32

    @abc.abstractmethod
    def get_variables(self):
        """Returns the variables used by this Agent."""


class RandomAgent(AgentBase):
    """Produces actions randomly, for exploration."""

    def __init__(self, hparams, name=None):
        super(RandomAgent, self).__init__(hparams, name=name)
        self._dist = tf.distributions.Dirichlet(tf.ones(self._num_actions))

    @property
    def state_size(self):
        return tf.TensorShape([0])

    @property
    def state_dtype(self):
        return tf.float32

    def observe(self, observation, state):
        return state

    def get_variables(self):
        return None

    def _build(self, input_, state):
        del input_  # Not used.
        batch_size = tf.shape(state)[0]
        return self._dist.sample(batch_size), state


class TrainableAgent(AgentBase):
    """Produces actions from a policy RNN."""

    def __init__(self, hparams, obs_encoder, name=None):
        super(TrainableAgent, self).__init__(hparams, name=name)
        self._agent_variables = None
        self._obs_encoder = obs_encoder
        with self._enter_variable_scope():
            self._policy_rnn = util.make_rnn(hparams, name="policy_rnn")
            self._project_act = util.make_mlp(
                hparams, layers=[hparams.game_action_space], name="policy_proj")

    @property
    def state_size(self):
        return dict(policy=self._policy_rnn.state_size,
                    action_logits=self._num_actions,
                    obs_enc=self._obs_encoder.output_size)

    @property
    def state_dtype(self):
        return snt.nest.map(lambda _: tf.float32, self.state_size)

    def get_variables(self):
        if self._agent_variables is None:
            raise ValueError("Agent variables haven't been constructed yet.")
        return self._agent_variables

    def observe(self, observation, state):
        obs_enc = self._obs_encoder(observation)
        rnn_state = state["policy"]
        hidden, rnn_state = self._policy_rnn(obs_enc, rnn_state)
        action_logits = self._project_act(hidden)
        if self._agent_variables is None:
            self._agent_variables = snt.nest.flatten(
                (self._policy_rnn.get_variables(),
                 self._project_act.get_variables()))
        if self._hparams.explore_temp > 0:
            dist = tf.contrib.distributions.ExpRelaxedOneHotCategorical(
                self._hparams.explore_temp,
                logits=action_logits)
            action_logits = dist.sample()
        return dict(policy=rnn_state,
                    action_logits=action_logits,
                    obs_enc=obs_enc)

    def _build(self, input_, state):
        if input_ is not None:
            raise ValueError("I don't know how to encode any inputs.")
        return state["action_logits"], state
