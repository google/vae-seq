# Copyright 2018 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple extension of VAE to a sequential setting.

Notation:
 - z_1:T are hidden states, random variables.
 - d_1:T, e_1:T, and f_1:T are deterministic RNN outputs.
 - x_1:T are the observed states.
 - c_1:T are per-timestep inputs.

        Generative model               Inference model
      =====================         =====================
      x_1               x_t             z_1        z_t
       ^                 ^               ^          ^
       |                 |               |          |
      d_1 ------------> d_t             f_1 <----- f_t
       ^                 ^               ^          ^
       |                 |               |          |
   [c_1, z_1]        [c_t, z_t]         e_1 -----> e_t
                                         ^          ^
                                         |          |
                                     [c_1, x_1] [c_t, x_t]
"""

import sonnet as snt
import tensorflow as tf

from .. import latent as latent_mod
from .. import util
from .. import vae_module

class IndependentSequence(vae_module.VAECore):
    """Implementation of a Sequential VAE with independent latent variables."""

    def __init__(self, hparams, obs_encoder, obs_decoder, name=None):
        super(IndependentSequence, self).__init__(
            hparams, obs_encoder, obs_decoder, name)
        with self._enter_variable_scope():
            self._d_core = util.make_rnn(hparams, name="d_core")
            self._e_core = util.make_rnn(hparams, name="e_core")
            self._f_core = util.make_rnn(hparams, name="f_core")
            self._q_z = latent_mod.LatentDecoder(hparams, name="latent_q")

    @property
    def state_size(self):
        return (self._d_core.state_size, self._q_z.event_size)

    def _build(self, input_, state):
        d_state, latent = state
        d_out, d_state = self._d_core(
            util.concat_features((input_, latent)), d_state)
        return self._obs_decoder(d_out), d_state

    def _next_state(self, d_state, event=None):
        del event  # Not used.
        batch_size = util.batch_size_from_nested_tensors(d_state)
        latent_dist = _latent_prior(self._hparams, batch_size)
        return (d_state, latent_dist)

    def _initial_state(self, batch_size):
        return self._next_state(
            self._d_core.initial_state(batch_size), event=None)

    def _infer_latents(self, inputs, observed):
        hparams = self._hparams
        batch_size = util.batch_size_from_nested_tensors(observed)
        enc_observed = snt.BatchApply(self._obs_encoder, n_dims=2)(observed)
        e_outs, _ = tf.nn.dynamic_rnn(
            self._e_core,
            util.concat_features((inputs, enc_observed)),
            initial_state=self._e_core.initial_state(batch_size))
        f_outs, _ = util.reverse_dynamic_rnn(
            self._f_core,
            e_outs,
            initial_state=self._f_core.initial_state(batch_size))
        q_zs = self._q_z.dist(
            snt.BatchApply(self._q_z, n_dims=2)(f_outs),
            name="q_zs")
        latents = q_zs.sample()
        p_zs = tf.contrib.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(latents),
            scale_diag=tf.ones_like(latents),
            name="p_zs")
        divs = util.calc_kl(hparams, latents, q_zs, p_zs)
        (_unused_d_outs, d_states), _ = tf.nn.dynamic_rnn(
            util.state_recording_rnn(self._d_core),
            util.concat_features((inputs, latents)),
            initial_state=self._d_core.initial_state(batch_size))
        return (d_states, latents), divs


def _latent_prior(hparams, batch_size):
    dims = tf.stack([batch_size, hparams.latent_size])
    loc = tf.zeros(dims)
    loc.set_shape([None, hparams.latent_size])
    scale_diag = tf.ones(dims)
    scale_diag.set_shape([None, hparams.latent_size])
    return tf.contrib.distributions.MultivariateNormalDiag(
        loc=loc, scale_diag=scale_diag, name="latent")
