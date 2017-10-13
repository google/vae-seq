"""Base classes for VAE implementations."""

import abc
import tensorflow as tf
import sonnet as snt

from .. import eval_core
from .. import gen_core
from .. import util


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

    @abc.abstractmethod
    def _init_submodules(self):
        """Called once to create latent and observation distcores."""

    @abc.abstractmethod
    def infer_latents(self, contexts, observed):
        """Returns a sequence of latent variabiables and their divergences."""

    def _build(self, *args, **kwargs):
        raise NotImplementedError("Please use member methods.")

    @util.lazy_property
    def eval_core(self):
        """An RNNCore is used to calculate log-probabilities of observations."""
        return eval_core.EvalCore(self.agent,
                                  self._latent_prior_distcore,
                                  self._observed_distcore)

    @util.lazy_property
    def gen_core(self):
        """A RNNCore is used to sample sequences."""
        return gen_core.GenCore(self.agent,
                                self._latent_prior_distcore,
                                self._observed_distcore,
                                with_obs_log_probs=False,
                                name="gen_core")

    @util.lazy_property
    def gen_log_probs_core(self):
        """A RNNCore is used to sample sequences."""
        return gen_core.GenCore(self.agent,
                                self._latent_prior_distcore,
                                self._observed_distcore,
                                with_obs_log_probs=True,
                                name="gen_log_probs_core")

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
        batch_size = util.batch_size_from_nested_tensors(observed)
        cell = self.observed_distcore.log_probs
        initial_state = cell.initial_state(batch_size)
        inputs = ((contexts, latents), observed)
        cell, inputs = util.add_support_for_scalar_rnn_inputs(cell, inputs)
        log_probs, _ = tf.nn.dynamic_rnn(
            cell, inputs,
            initial_state=initial_state,
            dtype=tf.float32)
        return log_probs
