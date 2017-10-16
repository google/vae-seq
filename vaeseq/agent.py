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

    def action(self, agent_input, state):
        """Optional method for interacting with an environment."""
        # Default to providing the context.
        return self.context(agent_input, state)

    @property
    def action_dtype(self):
        """The types of actions."""
        return self.context_dtype

    @property
    def action_size(self):
        """The sizes of the actions."""
        return self.context_size

    def get_variables(self):
        """Returns the variables to update during RL."""
        return None

    def rewards(self, observed):
        """Extract rewards from a sequence of observations."""
        del observed  # No rewards by default.
        return None

    def get_inputs(self, batch_size, sequence_size):
        """Creates an input tensor. Note that the sizes are suggestions."""
        # Use this if the agent doesn't take any external input.
        ret = tf.zeros([batch_size, sequence_size, 0], name="null_inputs")
        ret.set_shape([None, None, 0])
        return ret

    def contexts_for_static_observations(self, observed, agent_inputs=None,
                                         initial_state=None):
        """Generate contexts for a static sequence of observations."""
        batch_size = util.batch_size_from_nested_tensors(observed)
        if initial_state is None:
            initial_state = self.initial_state(batch_size)
        if agent_inputs is None:
            sequence_size = util.sequence_size_from_nested_tensors(observed)
            agent_inputs = self.get_inputs(batch_size, sequence_size)
            agent_inputs.set_shape(
                observed.get_shape()[:2].concatenate(
                    agent_inputs.get_shape()[2:]))

        def _step(input_obs, state):
            """Record the agent's context for the given observation."""
            agent_input, obs = input_obs
            context = self.context(agent_input, state)
            state = self.observe(agent_input, obs, state)
            return context, state

        cell = util.WrapRNNCore(_step, self.state_size, self.context_size)
        inputs = (agent_inputs, observed)
        cell, inputs = util.add_support_for_scalar_rnn_inputs(cell, inputs)
        contexts, _ = util.heterogeneous_dynamic_rnn(
            cell, inputs,
            initial_state=initial_state,
            output_dtypes=self.context_dtype)
        return contexts

    def run_environment(self, env, agent_inputs, env_initial_state=None,
                        agent_initial_state=None):
        """Generate contexts, actions, and observations from an Environment."""
        if env_initial_state is None or agent_initial_state is None:
            batch_size = tf.shape(agent_inputs)[0]
            if env_initial_state is None:
                env_initial_state = env.initial_state(batch_size)
            if agent_initial_state is None:
                agent_initial_state = self.initial_state(batch_size)
        initial_state = (agent_initial_state, env_initial_state)

        def _step(agent_input, state):
            """Manipulate the environment and record what happens."""
            agent_state, env_state = state
            context = self.context(agent_input, agent_state)
            action = self.action(agent_input, agent_state)
            obs, env_state = env(action, env_state)
            agent_state = self.observe(agent_input, obs, agent_state)
            return (context, action, obs), (agent_state, env_state)

        cell = util.WrapRNNCore(
            _step,
            state_size=(self.state_size, env.state_size),
            output_size=(self.context_size, self.action_size, env.output_size))
        output_dtype = (self.context_dtype, self.action_dtype, env.output_dtype)
        cell, agent_inputs = util.add_support_for_scalar_rnn_inputs(
            cell, agent_inputs)
        (contexts, actions, observations), _ = util.heterogeneous_dynamic_rnn(
            cell, agent_inputs,
            initial_state=initial_state,
            output_dtypes=output_dtype)
        return contexts, actions, observations


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
