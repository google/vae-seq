# -*- coding: utf-8 -*-
r""""SRNN as described in:

Marco Fraccaro, Søren Kaae Sønderby, Ulrich Paquet, Ole Winther.
Sequential Neural Models with Stochastic Layers.
https://arxiv.org/abs/1605.07571

Notation:
 - z_0:T are hidden states, random variables
 - d_1:T and e_1:T are deterministic RNN outputs
 - x_1:T are the observed states
 - c_1:T are the per-timestep contexts

    Generative model                Inference model
  =====================          =====================
 z_0 -> z_1 -----> z_t        z_0 -> z_1 ---------> z_t
  |      ^   |      ^                 ^              ^
  v      |   v      |                 |              |
 x_1 <-. |  x_t <-. |                 |              |
        \|         \|                e_1 <--------- e_t
         *          *               / ^            / ^
         |          |            x_1  |         x_t  |
        d_1 -----> d_t               d_1 ---------> d_t
         ^          ^                 ^              ^
         |          |                 |              |
        c_1        c_t               c_1            c_t
"""

import sonnet as snt
import tensorflow as tf

from .. import latent as latent_mod
from .. import util
from .. import vae_module

class SRNN(vae_module.VAECore):
    """Implementation of SRNN (see module description)."""

    def __init__(self, hparams, obs_encoder, obs_decoder, name=None):
        super(SRNN, self).__init__(hparams, obs_encoder, obs_decoder, name)
        with self._enter_variable_scope():
            self._d_core = util.make_rnn(hparams, name="d_core")
            self._e_core = util.make_rnn(hparams, name="e_core")
            self._latent_p = latent_mod.LatentDecoder(hparams, name="latent_p")
            self._latent_q = latent_mod.LatentDecoder(hparams, name="latent_q")

    @property
    def state_size(self):
        return (self._d_core.state_size, self._latent_p.event_size)

    def _build(self, context, state):
        d_state, latent = state
        d_out, d_state = self._d_core(util.concat_features(context), d_state)
        latent_params = self._latent_p(d_out, latent)
        return self._obs_decoder((d_out, latent)), (d_state, latent_params)

    def _next_state(self, state_arg, event=None):
        del event  # Not used.
        d_state, latent_params = state_arg
        return d_state, self._latent_p.dist(latent_params, name="latent")

    def _initial_state(self, batch_size):
        d_state = self._d_core.initial_state(batch_size)
        latent_input_sizes = (self._d_core.output_size,
                              self._latent_p.event_size)
        latent_inputs = snt.nest.map(
            lambda size: tf.zeros(
                [batch_size] + tf.TensorShape(size).as_list(),
                name="latent_input"),
            latent_input_sizes)
        latent_params = self._latent_p(latent_inputs)
        return self._next_state((d_state, latent_params), event=None)

    def infer_latents(self, contexts, observed):
        hparams = self._hparams
        batch_size = util.batch_size_from_nested_tensors(observed)
        d_initial, z_initial = self.initial_state(batch_size)
        (d_outs, d_states), _ = tf.nn.dynamic_rnn(
            util.state_recording_rnn(self._d_core),
            util.concat_features(contexts),
            initial_state=d_initial)
        enc_observed = snt.BatchApply(self._obs_encoder, n_dims=2)(observed)
        e_outs, _ = util.reverse_dynamic_rnn(
            self._e_core,
            util.concat_features((enc_observed, contexts)),
            initial_state=self._e_core.initial_state(batch_size))

        def _inf_step(d_e_outputs, prev_latent):
            """Iterate over d_1:T and e_1:T to produce z_1:T."""
            d_out, e_out = d_e_outputs
            p_z_params = self._latent_p(d_out, prev_latent)
            p_z = self._latent_p.dist(p_z_params)
            q_loc, q_scale = self._latent_q(e_out, prev_latent)
            if hparams.srnn_use_res_q:
                q_loc += p_z.loc
            q_z = self._latent_q.dist((q_loc, q_scale), name="q_z_dist")
            latent = q_z.sample()
            divergence = util.calc_kl(hparams, latent, q_z, p_z)
            return (latent, divergence), latent

        inf_core = util.WrapRNNCore(
            _inf_step,
            state_size=tf.TensorShape(hparams.latent_size),    # prev_latent
            output_size=(tf.TensorShape(hparams.latent_size),  # latent
                         tf.TensorShape([]),),                 # divergence
            name="inf_z_core")
        (latents, kls), _ = util.heterogeneous_dynamic_rnn(
            inf_core,
            (d_outs, e_outs),
            initial_state=z_initial,
            output_dtypes=(self._latent_q.event_dtype, tf.float32))
        return (d_states, latents), kls
