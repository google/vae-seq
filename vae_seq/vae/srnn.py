# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from . import latent
from .. import feedback_rnn_core
from .. import util


class SRNN(snt.AbstractModule):
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

    def __init__(self, hparams, obs_encoder, obs_decoder, name=None):
        super(SRNN, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder

    def _build(self, context, observed=None):
        hparams = self._hparams
        if observed is None:
            observed = util.dummy_observed(hparams)

        d_core = util.make_rnn(hparams, name='d_core')
        e_core = util.make_rnn(hparams, name='e_core')
        latent_p = latent.LatentDecoder(hparams, name='latent_p')
        latent_q = latent.LatentDecoder(hparams, name='latent_q')

        # Generative model.
        gen_core = feedback_rnn_core.FeedbackCore(
            d_core,
            _GenRNNFeedbackEncoder(hparams, self._obs_encoder),
            _GenRNNFeedbackDecoder(hparams, self._obs_decoder, latent_p))
        (d_initial, (z0, x0)) = gen_core_initial = gen_core.initial_state(
            hparams.batch_size, trainable=True)
        (gen_z, gen_x, gen_log_prob), _ = tf.nn.dynamic_rnn(
            gen_core,
            (context, observed),
            initial_state=gen_core_initial)

        # Bottom part of the inference model.
        enc_observed = snt.BatchApply(self._obs_encoder, n_dims=2)(observed)
        prev_enc_observed = tf.concat([
            tf.expand_dims(self._obs_encoder(x0), 1),
            enc_observed[:, :-1, :],
        ], axis=1)
        inf_d, _ = tf.nn.dynamic_rnn(
            d_core,
            tf.concat([prev_enc_observed, context], axis=2),
            initial_state=d_initial)
        inf_e, _ = util.reverse_dynamic_rnn(
            e_core,
            tf.concat([enc_observed, inf_d], axis=2),
            initial_state=e_core.initial_state(
                hparams.batch_size, trainable=True))

        # Top part of the inference model.
        inf_core = _InfCore(hparams, self._obs_decoder, latent_p, latent_q)
        (inf_z, inf_x, inf_log_prob, inf_kl), _ = tf.nn.dynamic_rnn(
            inf_core,
            (inf_d, inf_e, observed),
            initial_state=z0)

        return util.VAETensors(
            gen_z, gen_x, util.squeeze_sum(gen_log_prob),
            inf_z, inf_x, util.squeeze_sum(inf_log_prob),
            util.squeeze_sum(inf_kl),
        )


class _GenRNNFeedbackEncoder(snt.AbstractModule):
    def __init__(self, hparams, obs_encoder, name=None):
        super(_GenRNNFeedbackEncoder, self).__init__(
            name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_encoder = obs_encoder

    def _build(self, (context, observed), (prev_z, prev_x)):
        inner_input = tf.concat([self._obs_encoder(prev_x), context], axis=1)
        decoder_state = (prev_z, observed)
        return inner_input, decoder_state


class _GenRNNFeedbackDecoder(snt.AbstractModule):
    def __init__(self, hparams, obs_decoder, latent_p, name=None):
        super(_GenRNNFeedbackDecoder, self).__init__(
            name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_decoder = obs_decoder
        self._latent_p = latent_p

    @property
    def output_size(self):
        hparams = self._hparams
        outputs = (tf.TensorShape(hparams.state_size),   # z ~ p(z)
                   tf.TensorShape(hparams.obs_shape),    # x ~ p(x | z)
                   tf.TensorShape([1]))                  # log p(observed | z)
        feedback = (tf.TensorShape(hparams.state_size),  # prev_z
                    tf.TensorShape(hparams.obs_shape))   # prev_x
        return outputs, feedback

    def _build(self, d, (prev_z, observed)):
        hparams = self._hparams
        p_z = self._latent_p.dist(d, prev_z)
        z = p_z.sample()
        p_x = self._obs_decoder.dist(d, z)
        x = p_x.sample()
        log_prob = tf.expand_dims(p_x.log_prob(observed), axis=1)
        output = (z, x, log_prob)
        feedback = (z, x)
        return output, feedback


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
