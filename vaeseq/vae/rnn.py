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

from .. import util
from .. import vae_module

class RNN(vae_module.VAECore):
    """Implementation of an RNN as a sequential VAE where all latent
       variables are deterministic."""

    def __init__(self, hparams, obs_encoder, obs_decoder, name=None):
        super(RNN, self).__init__(hparams, obs_encoder, obs_decoder, name)
        with self._enter_variable_scope():
            self._d_core = util.make_rnn(hparams, name="d_core")

    @property
    def state_size(self):
        return self._d_core.state_size

    def _next_state(self, d_state, event=None):
        del event  # Not used.
        return d_state

    def _initial_state(self, batch_size):
        return self._d_core.initial_state(batch_size)

    def _build(self, context, d_state):
        d_out, d_state = self._d_core(util.concat_features(context), d_state)
        return self._obs_decoder(d_out), d_state

    def infer_latents(self, contexts, observed):
        """Because the RNN latent state is fully deterministic, there's no
           need to do two passes over the training data."""
        batch_size = util.batch_size_from_nested_tensors(observed)
        sequence_size = util.sequence_size_from_nested_tensors(observed)
        divs = tf.zeros([batch_size, sequence_size], name="divergences")
        return None, divs

    def really_infer_latents(self, contexts, observed):
        batch_size = util.batch_size_from_nested_tensors(observed)
        sequence_size = util.sequence_size_from_nested_tensors(observed)
        cell = util.state_recording_rnn(self._d_core)
        outputs_and_states, _ = tf.nn.dynamic_rnn(
            cell,
            util.concat_features(contexts),
            initial_state=cell.initial_state(batch_size))
        latents = outputs_and_states[1]
        divs = tf.zeros([batch_size, sequence_size], name="divergences")
        return latents, divs
