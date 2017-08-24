"""Agents take observations and provide a context for the next timestep."""

import abc
import sonnet as snt
import tensorflow as tf

from . import util


class Agent(snt.AbstractModule):
    """Base class for agents."""

    @property
    @abc.abstractmethod
    def context_size(self):
        return

    @property
    @abc.abstractmethod
    def state_size(self):
        return

    @property
    @abc.abstractmethod
    def state_dtype(self):
        return

    @abc.abstractmethod
    def initial_state(self, batch_size):
        return

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
    def state_size(self):
        return self.context_size

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
        context = agent.context(agent_input, state)
        state = agent.observe(agent_input, obs, state)
        return context, state

    contexts, _ = tf.nn.dynamic_rnn(
        util.WrapRNNCore(_step, agent.state_size, agent.context_size),
        (agent_inputs, observations),
        initial_state=initial_state,
        dtype=tf.float32)
    return contexts


def contexts_and_observations_from_environment(env, agent, agent_inputs):
    """Generate contexts and observations from an environment RNNCore."""
    batch_size = tf.shape(agent_inputs)[0]

    # tf.nn.dynamic_rnn doesn't support heterogeneous output types, so we
    # only output the contexts. The observations are written into a TensorArray
    # in the state.
    obs_ta = tf.TensorArray(env.output_dtype, size=tf.shape(agent_inputs)[1])
    initial_state = (0,  # step
                     obs_ta,
                     agent.initial_state(batch_size),
                     env.initial_state(batch_size),)

    def _step(agent_input, (step, obs_ta, agent_state, env_state)):
        context = agent.context(agent_input, agent_state)
        env_input = agent.env_input(agent_input, agent_state)
        obs, env_state = env(env_input, env_state)
        agent_state = agent.observe(agent_input, obs, agent_state)
        obs_ta = obs_ta.write(step, obs)
        return context, (step + 1, obs_ta, agent_state, env_state)

    contexts, state = tf.nn.dynamic_rnn(
        util.WrapRNNCore(
            _step,
            state_size=(None, None, agent.state_size, env.state_size),
            output_size=agent.context_size),
        agent_inputs,
        initial_state=initial_state,
        dtype=tf.float32)
    observations = util.rnn_aux_output_to_batch_major(
        state[1], agent_inputs)
    return contexts, observations
