# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from . import base
from . import latent
from .. import util

class SRNN(base.VAEBase):
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

    def _allocate(self):
        hparams = self._hparams
        self._d_core = util.make_rnn(hparams, name="d_core")
        self._e_core = util.make_rnn(hparams, name="e_core")
        self._latent_p = latent.LatentDecoder(hparams, name="latent_p")
        self._latent_q = latent.LatentDecoder(hparams, name="latent_q")
        self._z_distcore = LatentPrior(hparams, self._d_core, self._latent_p)
        self._x_distcore = ObsDist(hparams, self._d_core, self._obs_decoder)

    @property
    def latent_prior_distcore(self):
        return self._z_distcore

    @property
    def observed_distcore(self):
        return self._x_distcore

    def infer_latents(self, contexts, observed):
        hparams = self._hparams
        z_initial, d_initial = self._z_distcore.samples.initial_state(
            hparams.batch_size)
        ds, _ = tf.nn.dynamic_rnn(
            self._d_core,
            util.concat_features(contexts),
            initial_state=d_initial)
        enc_observed = snt.BatchApply(self._obs_encoder, n_dims=2)(observed)
        es, _ = util.reverse_dynamic_rnn(
            self._e_core,
            util.concat_features((enc_observed, contexts)),
            initial_state=self._e_core.initial_state(hparams.batch_size))

        def _inf_step((d, e), prev_z):
            p_z = self._latent_p.dist(d, prev_z)
            q_loc, q_scale = self._latent_q(e, prev_z)
            if hparams.srnn_use_res_q:
                q_loc += p_z.loc
            q_z = self._latent_q.output_dist((q_loc, q_scale), name="q_z_dist")
            z = q_z.sample()
            kl = util.calc_kl(hparams, z, q_z, p_z)
            return (z, kl), z
        inf_core = util.WrapRNNCore(
            _inf_step,
            state_size=tf.TensorShape(hparams.latent_size),    # prev_z
            output_size=(tf.TensorShape(hparams.latent_size),  # z
                         tf.TensorShape([]),),                 # divergence
            name="inf_z_core")
        (zs, kls), _ = tf.nn.dynamic_rnn(
            inf_core,
            (ds, es),
            initial_state=z_initial)
        return zs, kls


class ObsDist(base.DistCore):
    def __init__(self, hparams, d_core, obs_decoder, name=None):
        super(ObsDist, self).__init__(name or self.__class__.__name__)
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

    def _build_dist(self, (context, z), d_state):
        d, d_state = self._d_core(util.concat_features(context), d_state)
        return self._obs_decoder.dist(d, z), d_state

    def _next_state(self, d_state, event=None):
        return d_state


class LatentPrior(base.DistCore):
    def __init__(self, hparams, d_core, latent_p, name=None):
        super(LatentPrior, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._d_core = d_core
        self._latent_p = latent_p

    @property
    def state_size(self):
        return (tf.TensorShape(self._hparams.latent_size),  # prev_z
                self._d_core.state_size,)                   # d state

    @property
    def event_size(self):
        return tf.TensorShape(self._hparams.latent_size)

    @property
    def event_dtype(self):
        return self._latent_p.event_dtype

    def _build_dist(self, context, state):
        prev_z, d_state = state
        d, d_state = self._d_core(util.concat_features(context), d_state)
        return self._latent_p.dist(d, prev_z), d_state

    def _next_state(self, d_state, event=None):
        return (event, d_state)
