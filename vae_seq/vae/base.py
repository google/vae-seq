"""Base classes for VAE implementations."""

import abc
import tensorflow as tf
import sonnet as snt

from .. import agent as agent_mod
from .. import dist_module
from .. import util


class GenCore(snt.RNNCore):
    """An RNNCore that generates sequences of observed and latent variables."""

    def __init__(self, agent, latent_distcore, obs_distcore, name=None):
        super(GenCore, self).__init__(name=name)
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
        return (self._obs_distcore.event_size,     # sampled observations
                self._latent_distcore.event_size,  # sampled latents
                self._agent.state_size,)           # agent states

    @property
    def output_dtype(self):
        """Types of output tensors."""
        return (self._obs_distcore.event_dtype,
                self._latent_distcore.event_dtype,
                self._agent.state_dtype)

    def _build(self, input_, state):
        agent_state, latent_state, obs_state = state
        context = self._agent.context(input_, agent_state)
        latent, latent_state = self._latent_distcore.samples(
            context, latent_state)
        obs, obs_state = self._obs_distcore.samples(
            (context, latent), obs_state)
        output = (obs, latent, agent_state)
        agent_state = self._agent.observe(input_, obs, agent_state)
        state = (agent_state, latent_state, obs_state)
        return output, state

    def generate(self, agent_inputs, initial_state=None):
        """Generates sequences of observations, latents, and agent states."""
        if initial_state is None:
            batch_size = tf.shape(agent_inputs)[0]
            initial_state = self.initial_state(batch_size)
        return util.heterogeneous_dynamic_rnn(
            self,
            agent_inputs,
            initial_state=initial_state,
            output_dtypes=self.output_dtype)[0]


class VAEBase(snt.AbstractModule):
    """Base class for Sequential VAE implementations."""

    def __init__(self, agent, name=None):
        super(VAEBase, self).__init__(name=name)
        self._agent = agent
        self._latent_prior_distcore = None
        self._observed_distcore = None
        with self._enter_variable_scope():
            self._init_submodules()
            assert self._latent_prior_distcore is not None
            assert self._observed_distcore is not None
            self._gen_core = GenCore(self.agent, self._latent_prior_distcore,
                                     self._observed_distcore)

    @abc.abstractmethod
    def _init_submodules(self):
        """Called once to create latent and observation distcores."""

    @abc.abstractmethod
    def infer_latents(self, contexts, observed):
        """Returns a sequence of latent variabiables and their divergences."""

    def _build(self, *args, **kwargs):
        raise NotImplementedError("Please use member methods.")

    @property
    def gen_core(self):
        """A GenCore is used to sample sequences."""
        return self._gen_core

    @property
    def agent(self):
        """An Agent produces contexts from observations."""
        return self._agent

    @property
    def latent_prior_distcore(self):
        """A DistCore: context -> p(latent)."""
        return self._latent_prior_distcore

    @property
    def observed_distcore(self):
        """A DistCore: (context, latent) -> p(observed)."""
        return self._observed_distcore

    def log_prob_observed(self, contexts, latents, observed):
        """Evaluates log probabilities of a sequence of observations."""
        batch_size = tf.shape(observed)[0]
        log_probs_core = self.observed_distcore.log_probs
        initial_state = log_probs_core.initial_state(batch_size)
        log_probs, _ = tf.nn.dynamic_rnn(
            log_probs_core,
            ((contexts, latents), observed),
            initial_state=initial_state,
            dtype=tf.float32)
        return log_probs
