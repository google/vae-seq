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
      x_1        x_t        z_0 -> z_1 ---------> z_t
     / ^        / ^                 ^              ^
    |  |       |  |                 |              |
z_0-|>z_1 -----> z_t               e_1 <--------- e_t
    |  ^       |  ^               / ^           /  ^
     \ |        \ |            x_1  |        x_t   |
      d_1 -----> d_t               d_1 ---------> d_t
       ^          ^                 ^              ^
       |          |                 |              |
      c_1        c_t               c_1            c_t
"""

import sonnet as snt
import tensorflow as tf

from . import base
from . import latent as latent_mod
from .. import util

class SRNN(base.VAEBase):
    """Implementation of SRNN (see module description)."""

    def __init__(self, hparams, agent, obs_encoder, obs_decoder, name=None):
        self._hparams = hparams
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder
        super(SRNN, self).__init__(agent, name=name)

    def _init_submodules(self):
        hparams = self._hparams
        self._d_core = util.make_rnn(hparams, name="d_core")
        self._e_core = util.make_rnn(hparams, name="e_core")
        self._latent_p = latent_mod.LatentDecoder(hparams, name="latent_p")
        self._latent_q = latent_mod.LatentDecoder(hparams, name="latent_q")
        self._latent_prior_distcore = LatentPrior(
            hparams, self._d_core, self._latent_p)
        self._observed_distcore = ObsDist(
            hparams, self._d_core, self._obs_decoder)

    def infer_latents(self, contexts, observed):
        hparams = self._hparams
        batch_size = util.batch_size(hparams)
        z_initial, d_initial = self.latent_prior_distcore.samples.initial_state(
            batch_size)
        d_outs, _ = tf.nn.dynamic_rnn(
            self._d_core,
            util.concat_features(contexts),
            initial_state=d_initial)
        enc_observed = snt.BatchApply(self._obs_encoder, n_dims=2)(observed)
        e_outs, _ = util.reverse_dynamic_rnn(
            self._e_core,
            util.concat_features((enc_observed, contexts)),
            initial_state=self._e_core.initial_state(batch_size))

        def _inf_step((d_out, e_out), prev_latent):
            """Iterate over d_1:T and e_1:T to produce z_1:T."""
            p_z = self._latent_p.dist(d_out, prev_latent)
            q_loc, q_scale = self._latent_q(e_out, prev_latent)
            if hparams.srnn_use_res_q:
                q_loc += p_z.loc
            q_z = self._latent_q.output_dist((q_loc, q_scale), name="q_z_dist")
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
        return latents, kls


class ObsDist(base.DistCore):
    """DistCore for producing p(observation | context, latent)."""

    def __init__(self, hparams, d_core, obs_decoder, name=None):
        super(ObsDist, self).__init__(name=name)
        self._hparams = hparams
        self._d_core = d_core
        self._obs_decoder = obs_decoder

    @property
    def state_size(self):
        return self._d_core.state_size

    @property
    def event_size(self):
        return tf.TensorShape(self._hparams.obs_shape)

    @property
    def event_dtype(self):
        return self._obs_decoder.event_dtype

    def _build_dist(self, (context, latent), d_state):
        d_out, d_state = self._d_core(util.concat_features(context), d_state)
        return self._obs_decoder.dist(d_out, latent), d_state

    def _next_state(self, d_state, event=None):
        return d_state


class LatentPrior(base.DistCore):
    """DistCore that produces Normal latent variables."""

    def __init__(self, hparams, d_core, latent_p, name=None):
        super(LatentPrior, self).__init__(name=name)
        self._hparams = hparams
        self._d_core = d_core
        self._latent_p = latent_p

    @property
    def state_size(self):
        return (tf.TensorShape(self._hparams.latent_size),  # prev_latent
                self._d_core.state_size,)                   # d state

    @property
    def event_size(self):
        return tf.TensorShape(self._hparams.latent_size)

    @property
    def event_dtype(self):
        return self._latent_p.event_dtype

    def _build_dist(self, context, state):
        prev_latent, d_state = state
        d_out, d_state = self._d_core(util.concat_features(context), d_state)
        return self._latent_p.dist(d_out, prev_latent), d_state

    def _next_state(self, d_state, event=None):
        return (event, d_state)
