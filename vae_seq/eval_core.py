"""Module for evaluating the likelihood of a given sequence."""

import tensorflow as tf
import sonnet as snt

from . import agent as agent_mod
from . import dist_module
from . import util


class EvalCore(snt.RNNCore):
    """An RNNCore that outputs log-probabilities for the given observations.

    Note that the log-probabilities will be based on sampled latent states.
    """

    def __init__(self, agent, latent_distcore, obs_distcore, name=None):
        super(EvalCore, self).__init__(name=name)
        assert isinstance(agent, agent_mod.Agent)
        assert isinstance(latent_distcore, dist_module.DistCore)
        assert isinstance(obs_distcore, dist_module.DistCore)
        self._agent = agent
        self._latent_distcore = latent_distcore
        self._obs_distcore = obs_distcore

    @property
    def state_size(self):
        """Sizes of state tensors."""
        return (self._agent.state_size,            # agent state
                self._latent_distcore.state_size,  # latent core state
                self._obs_distcore.state_size,)    # observation core state

    def initial_state(self, batch_size):
        """Override default implementation to support heterogeneous dtypes."""
        # TODO: support trainable initial states.
        return (self._agent.initial_state(batch_size),
                self._latent_distcore.samples.initial_state(batch_size),
                self._obs_distcore.samples.initial_state(batch_size),)

    @property
    def output_size(self):
        """Sizes of output tensors."""
        return tf.TensorShape([])  # log_prob

    def _build(self, (input_, obs), state):
        agent_state, latent_state, obs_state = state
        context = self._agent.context(input_, agent_state)
        latent, latent_state = self._latent_distcore.samples(
            context, latent_state)
        output, obs_state = self._obs_distcore.log_probs(
            ((context, latent), obs), obs_state)
        agent_state = self._agent.observe(input_, obs, agent_state)
        state = (agent_state, latent_state, obs_state)
        return output, state

    def log_probs(self, agent_inputs, observed, initial_state=None):
        """Compute a monte-carlo estimate of log-prob(observed)."""
        if initial_state is None:
            batch_size = tf.shape(agent_inputs)[0]
            initial_state = self.initial_state(batch_size)
        cell = self
        inputs = (agent_inputs, observed)
        cell, inputs = util.add_support_for_scalar_rnn_inputs(cell, inputs)
        return tf.nn.dynamic_rnn(
            cell, inputs,
            initial_state=initial_state,
            dtype=tf.float32)[0]
