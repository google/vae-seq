"""Module for generating observations from a trained generative model."""

import tensorflow as tf
import sonnet as snt

from . import agent as agent_mod
from . import dist_module
from . import util


class GenCore(snt.RNNCore):
    """An RNNCore that generates sequences of observed and latent variables."""

    def __init__(self, agent, latent_distcore, obs_distcore,
                 with_obs_log_probs=False, name=None):
        super(GenCore, self).__init__(name=name)
        assert isinstance(agent, agent_mod.Agent)
        assert isinstance(latent_distcore, dist_module.DistCore)
        assert isinstance(obs_distcore, dist_module.DistCore)
        self._agent = agent
        self._latent_distcore = latent_distcore
        self._obs_distcore = obs_distcore
        self._with_obs_log_probs = with_obs_log_probs

    @property
    def agent(self):
        return self._agent

    @property
    def state_size(self):
        """Sizes of state tensors."""
        return (self.agent.state_size,             # agent state
                self._latent_distcore.state_size,  # latent core state
                self._obs_distcore.state_size,)    # observation core state

    def initial_state(self, batch_size):
        """Override default implementation to support heterogeneous dtypes."""
        # TODO: support trainable initial states.
        return (self.agent.initial_state(batch_size),
                self._latent_distcore.samples.initial_state(batch_size),
                self._obs_distcore.samples.initial_state(batch_size),)

    @property
    def output_size(self):
        """Sizes of output tensors."""
        obs_event_size = self._obs_distcore.event_size
        if self._with_obs_log_probs:
            obs_event_size = (obs_event_size, tf.TensorShape([]))
        return (obs_event_size,  # sampled observations (and log-probs)
                self._latent_distcore.event_size,  # sampled latents
                self.agent.state_size,)            # agent states

    @property
    def output_dtype(self):
        """Types of output tensors."""
        obs_event_dtype = self._obs_distcore.event_dtype
        if self._with_obs_log_probs:
            obs_event_dtype = (obs_event_dtype, tf.float32)
        return (obs_event_dtype,
                self._latent_distcore.event_dtype,
                self.agent.state_dtype)

    def _build(self, input_, state):
        agent_state, latent_state, obs_state = state
        context = self.agent.context(input_, agent_state)
        latent, latent_state = self._latent_distcore.next_sample(
            context, latent_state)
        obs, obs_state = self._obs_distcore.next_sample(
            (context, latent), obs_state,
            with_log_prob=self._with_obs_log_probs)
        output = (obs, latent, agent_state)
        obs_no_log_prob = obs[0] if self._with_obs_log_probs else obs
        agent_state = self.agent.observe(input_, obs_no_log_prob, agent_state)
        state = (agent_state, latent_state, obs_state)
        return output, state

    def generate(self, agent_inputs, initial_state=None):
        """Generates sequences of observations, latents, and agent states."""
        if initial_state is None:
            batch_size = util.batch_size_from_nested_tensors(agent_inputs)
            initial_state = self.initial_state(batch_size)
        cell = self
        cell, agent_inputs = util.add_support_for_scalar_rnn_inputs(
            cell, agent_inputs)
        return util.heterogeneous_dynamic_rnn(
            cell, agent_inputs,
            initial_state=initial_state,
            output_dtypes=self.output_dtype)[0]
