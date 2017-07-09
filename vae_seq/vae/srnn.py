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
       [x_0, c_1] [x_t-1, c_t]     [x_0, c_1]    [x_t-1, c_t]
    """

    def _build_vae(self, context, observed, enc_observed):
        hparams = self._hparams
        obs_encoder = self._obs_encoder
        obs_decoder = self._obs_decoder

        d_core = util.make_rnn(hparams, name='d_core')
        e_core = util.make_rnn(hparams, name='e_core')
        latent_p = latent.LatentDecoder(hparams, name='latent_p')
        latent_q = latent.LatentDecoder(hparams, name='latent_q')

        # Generative model.
        gen_core = _GenCore(hparams, d_core, obs_encoder, obs_decoder, latent_p)
        (d_initial, z0, x0) = gen_initial = gen_core.initial_state(
            hparams.batch_size, trainable=True)
        (gen_z, gen_x, gen_log_probs), _ = tf.nn.dynamic_rnn(
            gen_core,
            (context, observed),
            initial_state=gen_initial)

        # Bottom part of the inference model.
        prev_enc_observed = tf.concat([
            tf.expand_dims(obs_encoder(x0), 1),
            enc_observed[:, :-1, :],
        ], axis=1)
        inf_d, _ = tf.nn.dynamic_rnn(
            d_core,
            util.concat_features([prev_enc_observed, context]),
            initial_state=d_initial)
        inf_e, _ = util.reverse_dynamic_rnn(
            e_core,
            util.concat_features([enc_observed, inf_d]),
            initial_state=e_core.initial_state(
                hparams.batch_size, trainable=True))

        # Top part of the inference model.
        inf_core = _InfCore(hparams, obs_decoder, latent_p, latent_q)
        (inf_z, inf_x, inf_log_probs, inf_kls), _ = tf.nn.dynamic_rnn(
            inf_core,
            (inf_d, inf_e, observed),
            initial_state=z0)

        return util.VAETensors(
            gen_z, gen_x, util.squeeze_sum(gen_log_probs),
            inf_z, inf_x, util.squeeze_sum(inf_log_probs),
            util.squeeze_sum(inf_kls),
        )


class _GenCore(snt.RNNCore):
    def __init__(self, hparams, d_core, obs_encoder, obs_decoder, latent_p,
                 name=None):
        super(_GenCore, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._d_core = d_core
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder
        self._latent_p = latent_p

    @property
    def state_size(self):
        hparams = self._hparams
        return (self._d_core.state_size,
                tf.TensorShape(hparams.state_size),  # prev_z
                tf.TensorShape(hparams.obs_shape))   # prev_x

    @property
    def output_size(self):
        hparams = self._hparams
        return (tf.TensorShape(hparams.state_size),  # z ~ p(z)
                tf.TensorShape(hparams.obs_shape),   # x ~ p(x | z)
                tf.TensorShape([1]))                 # log p(x = observed | z)

    def _build(self, (context, observed), (d_state, prev_z, prev_x)):
        hparams = self._hparams
        d_input = util.concat_features([self._obs_encoder(prev_x), context])
        d, d_state = self._d_core(d_input, d_state)
        p_z = self._latent_p.dist(d, prev_z)
        z = p_z.sample()
        p_x = self._obs_decoder.dist(d, z)
        x = p_x.sample()
        log_prob = tf.expand_dims(p_x.log_prob(observed), axis=1)
        return (z, x, log_prob), (d_state, z, x)


class _InfCore(snt.RNNCore):
    """The top layer of the inference model."""
    def __init__(self, hparams, obs_decoder, latent_p, latent_q, name=None):
        super(_InfCore, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_decoder = obs_decoder
        self._latent_p = latent_p
        self._latent_q = latent_q

    @property
    def state_size(self):
        return tf.TensorShape(self._hparams.state_size)  # prev_z

    @property
    def output_size(self):
        hparams = self._hparams
        return (tf.TensorShape(hparams.state_size),  # z ~ q(z | d, e, prev_z)
                tf.TensorShape(hparams.obs_shape),   # x ~ p(x | z)
                tf.TensorShape([1]),                 # log prob(observed | z)
                tf.TensorShape([1]))                 # kl(q(z) || p(z))

    def _build(self, (d, e, observed), prev_z):
        hparams = self._hparams
        p_z = self._latent_p.dist(d, prev_z)
        q_loc, q_scale = self._latent_q(e, prev_z)
        if hparams.srnn_use_res_q:
            q_loc += p_z.loc
        q_z = self._latent_q.output_dist((q_loc, q_scale), name='q_z_dist')
        z = q_z.sample()
        p_x = self._obs_decoder.dist(d, z)
        x = p_x.sample()
        kl = tf.expand_dims(util.calc_kl(hparams, z, q_z, p_z), axis=1)
        log_prob = tf.expand_dims(p_x.log_prob(observed), axis=1)
        output = (z, x, log_prob, kl)
        return output, z
