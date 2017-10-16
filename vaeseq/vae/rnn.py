"""We can view an RNN as a VAE with no latent variables:

Notation:
 - d_1:T are the (deterministic) RNN outputs.
 - x_1:T are the observed states.
 - c_1:T are per-timestep contexts.

        Generative model
      =====================
      x_1               x_t
       ^                 ^
       |                 |
      d_1 ------------> d_t
       ^                 ^
       |                 |
      c_1               c_t
"""

import tensorflow as tf
from tensorflow.contrib import distributions

from . import base
from .. import dist_module
from .. import util

class RNN(base.VAEBase):
    """Implementation of an RNN as a fake sequential VAE."""

    def __init__(self, hparams, agent, obs_encoder, obs_decoder, name=None):
        self._hparams = hparams
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder
        super(RNN, self).__init__(agent, name=name)

    def _init_submodules(self):
        hparams = self._hparams
        self._d_core = util.make_rnn(hparams, name="d_core")
        self._latent_prior_distcore = NoLatents(hparams)
        self._observed_distcore = ObsDist(self._d_core, self._obs_decoder)

    def infer_latents(self, contexts, observed):
        del contexts  # Not used.
        batch_size = util.batch_size_from_nested_tensors(observed)
        sequence_size = util.sequence_size_from_nested_tensors(observed)
        latents = tf.zeros([batch_size, sequence_size, 0])
        divs = tf.zeros([batch_size, sequence_size])
        return latents, divs


class ObsDist(dist_module.DistCore):
    """DistCore for producing p(observation | context, latent)."""

    def __init__(self, d_core, obs_decoder, name=None):
        super(ObsDist, self).__init__(name=name)
        self._d_core = d_core
        self._obs_decoder = obs_decoder

    @property
    def state_size(self):
        return self._d_core.state_size

    @property
    def event_size(self):
        return self._obs_decoder.event_size

    @property
    def event_dtype(self):
        return self._obs_decoder.event_dtype

    def dist(self, params, name=None):
        return self._obs_decoder.dist(params, name=name)

    def _next_state(self, d_state, event=None):
        del event  # Not used.
        return d_state

    def _build(self, inputs, d_state):
        context, latent = inputs
        del latent  # The latent variable is empty (has zero size).
        d_out, d_state = self._d_core(util.concat_features(context), d_state)
        return self._obs_decoder(d_out), d_state


class NoLatents(dist_module.DistCore):
    """DistCore that samples an empty latent state."""

    def __init__(self, hparams, name=None):
        super(NoLatents, self).__init__(name=name)
        self._hparams = hparams

    @property
    def state_size(self):
        return ()

    @property
    def event_size(self):
        return tf.TensorShape([0])

    @property
    def event_dtype(self):
        return tf.float32

    def dist(self, batch_size, name=None):
        null_events = tf.zeros([batch_size, 0], dtype=self.event_dtype)
        null_events.set_shape([None, 0])
        return tf.contrib.distributions.VectorDeterministic(
            null_events, name=name or self.module_name + "_dist")

    def _next_state(self, state_arg, event=None):
        del state_arg  # No state needed.
        return ()

    def _build(self, context, state):
        del state  # No state needed.
        batch_size = util.batch_size_from_nested_tensors(context)
        return batch_size, ()
