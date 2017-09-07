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

"""Agents take observations and provide a context for the next timestep."""

import abc
import sonnet as snt
import tensorflow as tf

from . import util


class Agent(snt.AbstractModule):
    """Base class for agents."""

    @abc.abstractproperty
    def context_size(self):
        """The non-batch sizes of Tensors returned by self.context."""

    @abc.abstractproperty
    def context_dtype(self):
        """The types of Tensors returned by self.context."""

    @abc.abstractproperty
    def state_size(self):
        """The non-batch sizes of this Agent's state Tensors."""

    @abc.abstractproperty
    def state_dtype(self):
        """The types of Tensors in this Agent's state."""

    @abc.abstractmethod
    def initial_state(self, batch_size):
        """Returns an initial state tuple."""

    @abc.abstractmethod
    def observe(self, agent_input, observation, state):
        """Returns the updated state."""

    @abc.abstractmethod
    def context(self, agent_input, state):
        """Returns a context for the current time step."""

    def _build(self, *args, **kwargs):
        raise NotImplementedError("Please use member methods.")

    def env_input(self, agent_input, state):
        """Optional method to generate training data from an environment."""
        # Default to providing the context.
        return self.context(agent_input, state)


class Environment(snt.RNNCore):
    """An Environment is an RNN that takes contexts and returns observations."""

    @abc.abstractproperty
    def output_dtype(self):
        """The types of observation Tensors."""


class EncodeObsAgent(Agent):
    """Simple agent that encodes the previous observation as context."""

    def __init__(self, obs_encoder, name=None):
        super(EncodeObsAgent, self).__init__(name=name)
        self._obs_encoder = obs_encoder

    @property
    def context_size(self):
        # Encoding of the last observation.
        return self._obs_encoder.output_size

    @property
    def context_dtype(self):
        return tf.float32

    @property
    def state_size(self):
        # See context_size.
        return self._obs_encoder.output_size

    @property
    def state_dtype(self):
        return tf.float32

    def initial_state(self, batch_size):
        return tf.zeros([batch_size] +
                        tf.TensorShape(self._obs_encoder.output_size).as_list())

    def observe(self, agent_input, observation, state):
        return self._obs_encoder(observation)

    def context(self, agent_input, state):
        return state


def null_inputs(batch_size, sequence_size, dtype=tf.float32, name=None):
    """Use this if the agent doesn't take any external input."""
    return tf.zeros([batch_size, sequence_size, 0], dtype=dtype, name=name)


def contexts_for_static_observations(observations, agent, agent_inputs):
    """Generate contexts for a static sequence of observations."""
    batch_size = tf.shape(agent_inputs)[0]
    initial_state = agent.initial_state(batch_size)

    def _step((agent_input, obs), state):
        """Record the agent's context for the given observation."""
        context = agent.context(agent_input, state)
        state = agent.observe(agent_input, obs, state)
        return context, state

    contexts, _ = util.heterogeneous_dynamic_rnn(
        util.WrapRNNCore(_step, agent.state_size, agent.context_size),
        (agent_inputs, observations),
        initial_state=initial_state,
        output_dtypes=agent.context_dtype)
    return contexts


def contexts_and_observations_from_environment(env, agent, agent_inputs):
    """Generate contexts and observations from an environment RNNCore."""
    batch_size = tf.shape(agent_inputs)[0]
    initial_state = (agent.initial_state(batch_size),
                     env.initial_state(batch_size))

    def _step(agent_input, (agent_state, env_state)):
        """Have the agent manipulate the environment and record what happens."""
        context = agent.context(agent_input, agent_state)
        env_input = agent.env_input(agent_input, agent_state)
        obs, env_state = env(env_input, env_state)
        agent_state = agent.observe(agent_input, obs, agent_state)
        return (context, obs), (agent_state, env_state)

    (contexts, observations), _ = util.heterogeneous_dynamic_rnn(
        util.WrapRNNCore(
            _step,
            state_size=(agent.state_size, env.state_size),
            output_size=(agent.context_size, env.output_size)),
        agent_inputs,
        initial_state=initial_state,
        output_dtypes=(agent.context_dtype, env.output_dtype))
    return contexts, observations
