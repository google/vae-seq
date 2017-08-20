"""Base classes for VAE implementations."""

import abc
import tensorflow as tf
import sonnet as snt

from .. import agent as agent_mod
from .. import util

class DistCore(snt.AbstractModule):
    """Like an RNNCore, but outputs distributions."""

    @property
    @abc.abstractmethod
    def state_size(self):
        """Returns the non-batched sizes of Tensors returned from next_state."""

    @property
    @abc.abstractmethod
    def event_size(self):
        """Returns the output distribution event sizes."""

    @property
    @abc.abstractmethod
    def event_dtype(self):
        """Returns the output distribution event dtypes."""

    @property
    def samples(self):
        """Returns an RNNCore that outputs samples."""
        def _step(input_, state):
            """Samples from the distribution at each time step."""
            dist, state_arg = self(input_, state)
            event = dist.sample()
            state = self._next_state(state_arg, event)
            return event, state
        return util.WrapRNNCore(
            _step,
            self.state_size,
            output_size=self.event_size,
            name=self.module_name + "/Samples")

    @property
    def log_probs(self):
        """Returns an RNNCore that outputs log-probabilities."""
        def _step((input_, observed), state):
            """Calculates the log-probability of the observed event."""
            dist, state_arg = self(input_, state)
            state = self._next_state(state_arg, observed)
            return dist.log_prob(observed), state
        return util.WrapRNNCore(
            _step,
            self.state_size,
            output_size=tf.TensorShape([]),
            name=self.module_name + "/LogProbs")

    @abc.abstractmethod
    def _build_dist(self, input_, state):
        """Returns a distribution and a state_arg"""
        return

    @abc.abstractmethod
    def _next_state(self, state_arg, event=None):
        """Produces the next state given a state_arg and event.
        NOTE: this function shouldn't allocate variables."""
        return

    def _build(self, input_, state):
        ret, state_arg = self._build_dist(input_, state)
        assert all(isinstance(out, tf.contrib.distributions.Distribution)
                   for out in snt.nest.flatten(ret))
        return ret, state_arg


class GenCore(snt.RNNCore):
    """An RNNCore that generates sequences of observed and latent variables."""

    def __init__(self, agent, latent_distcore, obs_distcore, name=None):
        super(GenCore, self).__init__(name or self.__class__.__name__)
        assert isinstance(agent, agent_mod.Agent)
        assert isinstance(latent_distcore, DistCore)
        assert isinstance(obs_distcore, DistCore)
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


class VAEBase(snt.AbstractModule):
    """Base class for Sequential VAE implementations."""

    def __init__(self, agent, name=None):
        super(VAEBase, self).__init__(name or self.__class__.__name__)
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
