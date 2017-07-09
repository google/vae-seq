import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions

from . import base
from . import latent
from . import independent_sequence_distribution as iseq_dist
from .. import util

class IndependentSequenceVAE(base.VAEBase):
    """Simple extension of VAE to a sequential setting.

    Notation:
     - z_1:T are hidden states, random variables.
     - d_1:T, e_1:T, and f_1:T are deterministic RNN outputs.
     - x_1:T are the observed states.
     - c_1:T are per-timestep contexts.

            Generative model               Inference model
          =====================         =====================
          x_1               x_t             z_1        z_t
           ^                 ^               ^          ^
           |                 |               |          |
          d_1 ------------> d_t             f_1 <----- f_t
           ^                 ^               ^          ^
           |                 |               |          |
    [x_0, c_1, z_1] [x_t-1, c_t, z_t]       e_1 -----> e_t
                                             ^          ^
                                             |          |
                                         [c_1, x_1] [c_t, x_t]
    """

    def _prior_latent(self):
        """Prior distribution over the latent variables."""
        hparams = self._hparams
        dims = [hparams.batch_size, hparams.sequence_size, hparams.state_size]
        return iseq_dist.IndependentSequence(
            distributions.MultivariateNormalDiag(
                loc=tf.zeros(dims),
                scale_diag=tf.ones(dims),
                name='p_z_dist'))

    def _inferred_latent(self, context, enc_observed):
        """Variational distribution over the latent variables."""
        hparams = self._hparams
        inputs = util.concat_features([context, enc_observed])
        fwd_core = util.make_rnn(hparams, name='inf_fwd_core')
        inputs, _ = tf.nn.dynamic_rnn(
            fwd_core, inputs,
            initial_state=fwd_core.initial_state(
                hparams.batch_size, trainable=True))
        bwd_core = util.make_rnn(hparams, name='inf_bwd_core')
        inputs, _ = util.reverse_dynamic_rnn(
            bwd_core, inputs,
            initial_state=bwd_core.initial_state(
                hparams.batch_size, trainable=True))
        inputs = util.activation(hparams)(inputs)
        latent_dec = latent.LatentDecoder(hparams, name='q_z')
        return iseq_dist.IndependentSequence(
            latent_dec.output_dist(
                snt.BatchApply(latent_dec, n_dims=2)(inputs),
                name='q_z_dist'))

    def _build_vae(self, context, observed, enc_observed):
        hparams = self._hparams
        obs_encoder = self._obs_encoder
        obs_decoder = self._obs_decoder

        d_core = util.make_rnn(hparams, name='d_core')
        p_z = self._prior_latent()
        q_z = self._inferred_latent(context, enc_observed)

        # Generative model.
        gen_z = p_z.sample()
        gen_core = _GenCore(hparams, d_core, obs_encoder, obs_decoder)
        (d_initial, x0) = gen_initial = gen_core.initial_state(
            hparams.batch_size, trainable=True)
        (gen_x, gen_log_probs), _ = tf.nn.dynamic_rnn(
            gen_core, (context, gen_z, observed), initial_state=gen_initial)
        gen_log_prob = util.squeeze_sum(gen_log_probs)

        # Inference model.
        inf_z = q_z.sample()
        prev_enc_observed = tf.concat([
            tf.expand_dims(obs_encoder(x0), 1),
            enc_observed[:, :-1, :],
        ], axis=1)
        inf_d, _ = tf.nn.dynamic_rnn(
            d_core,
            util.concat_features([prev_enc_observed, context, inf_z]),
            initial_state=d_initial)
        inf_p_x = iseq_dist.IndependentSequence(
            obs_decoder.output_dist(
                snt.BatchApply(obs_decoder, n_dims=2)(inf_d),
                name='p_x_dist'))
        inf_x = inf_p_x.sample()
        inf_log_prob = inf_p_x.log_prob(observed)
        inf_kl = util.calc_kl(hparams, inf_z, q_z, p_z)

        return util.VAETensors(
            gen_z, gen_x, gen_log_prob,
            inf_z, inf_x, inf_log_prob, inf_kl)


class _GenCore(snt.RNNCore):
    def __init__(self, hparams, d_core, obs_encoder, obs_decoder, name=None):
        super(_GenCore, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._d_core = d_core
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder

    @property
    def state_size(self):
        hparams = self._hparams
        return (self._d_core.state_size,
                tf.TensorShape(hparams.obs_shape))  # prev_x

    @property
    def output_size(self):
        hparams = self._hparams
        return (tf.TensorShape(hparams.obs_shape),  # x ~ p(x | z)
                tf.TensorShape([1]))                # log p(x = observed | z)

    def _build(self, (context, gen_z, observed), (d_state, prev_sample)):
        enc_prev_sample = self._obs_encoder(prev_sample)
        d_input = util.concat_features([enc_prev_sample, context, gen_z])
        gen_d, d_state = self._d_core(d_input, d_state)
        gen_p_x = self._obs_decoder.dist(gen_d)
        sample = gen_p_x.sample()
        log_prob = tf.expand_dims(gen_p_x.log_prob(observed), 1)
        return (sample, log_prob), (d_state, sample)
