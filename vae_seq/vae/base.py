import abc
import collections
import tensorflow as tf
import sonnet as snt

from .. import agent as agent_mod
from .. import util

class DistCore(snt.AbstractModule):
    """Like an RNNCore, but outputs distributions."""

    @property
    @abc.abstractmethod
    def state_size(self):
        return

    @property
    @abc.abstractmethod
    def event_size(self):
        return

    @property
    @abc.abstractmethod
    def event_dtype(self):
        return

    @property
    def samples(self):
        def _step(input_, state):
            dist, state_arg = self(input_, state)
            event = dist.sample()
            state = self._next_state(state_arg, event)
            return event, state
        return util.WrapRNNCore(
            _step,
            self.state_size,
            self.event_size,
            name=self.module_name + "/Samples")

    @property
    def log_probs(self):
        def _step((input_, observed), state):
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

    def __init__(self, agent, z_distcore, x_distcore, name=None):
        super(GenCore, self).__init__(name or self.__class__.__name__)
        assert isinstance(agent, agent_mod.Agent)
        assert isinstance(z_distcore, DistCore)
        assert isinstance(x_distcore, DistCore)
        self._agent = agent
        self._z_distcore = z_distcore
        self._x_distcore = x_distcore

    @property
    def state_size(self):
        return (self._agent.state_size,         # agent state
                self._z_distcore.state_size,    # latent core state
                self._x_distcore.state_size,)   # observation core state

    def initial_state(self, batch_size):
        """Override default implementation to support heterogeneous dtypes."""
        # TODO: support trainable initial states.
        return (self._agent.initial_state(batch_size),
                self._z_distcore.samples.initial_state(batch_size),
                self._x_distcore.samples.initial_state(batch_size),)

    @property
    def output_size(self):
        return (self._x_distcore.event_size,  # sampled observations
                self._z_distcore.event_size,  # sampled latents
                self._agent.state_size,)      # agent states

    @property
    def output_dtype(self):
        return (self._x_distcore.event_dtype,
                self._z_distcore.event_dtype,
                self._agent.state_dtype)

    def _build(self, input_, state):
        agent_state, z_dist_state, x_dist_state = state
        context = self._agent.context(input_, agent_state)
        z, z_dist_state = self._z_distcore.samples(context, z_dist_state)
        x, x_dist_state = self._x_distcore.samples((context, z), x_dist_state)
        agent_state = self._agent.observe(input_, x, agent_state)
        state = (agent_state, z_dist_state, x_dist_state)
        return (x, z, agent_state), state


class VAEBase(snt.AbstractModule):
    """Base class for Sequential VAE implementations."""

    def __init__(self, hparams, agent, obs_encoder, obs_decoder, name=None):
        super(VAEBase, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._agent = agent
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder
        with self._enter_variable_scope():
            self._allocate()
            self._gen_core = GenCore(
                agent, self.latent_prior_distcore, self.observed_distcore)

    def _build(self, *args, **kwargs):
        raise NotImplementedError("Please use member methods.")

    @property
    def agent(self):
        return self._agent

    @property
    def gen_core(self):
        return self._gen_core

    def log_prob_observed(self, contexts, latents, observed):
        batch_size = tf.shape(observed)[0]
        log_probs_core = self.observed_distcore.log_probs
        initial_state = log_probs_core.initial_state(batch_size)
        log_probs, _ = tf.nn.dynamic_rnn(
            log_probs_core,
            ((contexts, latents), observed),
            initial_state=initial_state,
            dtype=tf.float32)
        return log_probs

    @abc.abstractmethod
    def _allocate(self):
        """Allocate modules to implement the methods below."""

    @property
    @abc.abstractmethod
    def latent_prior_distcore(self):
        """Returns a DistCore: context -> p(latent)."""

    @property
    @abc.abstractmethod
    def observed_distcore(self):
        """Returns a DistCore: (context, latent) -> p(observed)."""

    @abc.abstractmethod
    def infer_latents(self, contexts, observed):
        """Returns a sequence of latent variabiables and their divergences."""

