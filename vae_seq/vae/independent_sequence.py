import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions

from . import latent
from .. import feedback_rnn_core
from .. import util

class IndependentSequenceVAE(snt.AbstractModule):
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

    def __init__(self, hparams, obs_encoder, obs_decoder, name=None):
        super(IndependentSequenceVAE, self).__init__(
            name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder

    def _build(self, context, observed=None):
        hparams = self._hparams
        if observed is None:
            observed = util.dummy_observed(hparams)

        gen_rnn = _GenRNN(hparams, self._obs_encoder, self._obs_decoder)
        inf_rnn = _InfRNN(hparams, self._obs_encoder)

        # Generative model.
        p_z = _PriorState(hparams)
        gen_z = p_z.sample()
        gen_x, gen_log_prob = gen_rnn(gen_z, context, observed, feedback=True)

        # Inference model.
        q_z = inf_rnn(context, observed)
        inf_z = q_z.sample()
        inf_x, inf_log_prob = gen_rnn(inf_z, context, observed, feedback=False)
        inf_kl = util.calc_kl(hparams, inf_z, q_z, p_z)

        return util.VAETensors(
            gen_z, gen_x, gen_log_prob,
            inf_z, inf_x, inf_log_prob, inf_kl)


def _PriorState(hparams):
    """Prior distribution over the State."""
    dims = [hparams.batch_size, hparams.sequence_size, hparams.state_size]
    return IndependentSequence(
        distributions.MultivariateNormalDiag(
            loc=tf.zeros(dims),
            scale_diag=tf.ones(dims)))


class _GenRNN(snt.AbstractModule):
    """State -> samples, log_prob(observed)."""
    def __init__(self, hparams, obs_encoder, obs_decoder, name=None):
        super(_GenRNN, self).__init__(
            name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder

    def _build(self, z, context, observed, feedback=False):
        hparams = self._hparams
        core = feedback_rnn_core.FeedbackCore(
            util.make_rnn(hparams, name='gen_core'),
            _GenRNNFeedbackEncoder(hparams, self._obs_encoder),
            _GenRNNFeedbackDecoder(hparams, self._obs_decoder, feedback=feedback))
        (samples, log_probs), _ = tf.nn.dynamic_rnn(
            core, (context, z, observed),
            initial_state=core.initial_state(hparams.batch_size, trainable=True))
        return samples, util.squeeze_sum(log_probs)


class _GenRNNFeedbackEncoder(snt.AbstractModule):
    def __init__(self, hparams, obs_encoder, name=None):
        super(_GenRNNFeedbackEncoder, self).__init__(
            name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_encoder = obs_encoder

    def _build(self, (context, z, observed), feedback):
        inner_input = tf.concat([self._obs_encoder(feedback), context, z], axis=1)
        decoder_state = observed
        return inner_input, decoder_state


class _GenRNNFeedbackDecoder(snt.AbstractModule):
    def __init__(self, hparams, obs_decoder, feedback, name=None):
        super(_GenRNNFeedbackDecoder, self).__init__(
            name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_decoder = obs_decoder
        self._feedback = feedback

    @property
    def output_size(self):
        hparams = self._hparams
        outputs = (tf.TensorShape(hparams.obs_shape),  # sample
                   tf.TensorShape([1]))                # log_prob(observed)
        feedback = tf.TensorShape(hparams.obs_shape)
        return outputs, feedback

    def _build(self, inner_output, observed):
        p_x = self._obs_decoder.dist(inner_output)
        sample = p_x.sample()
        log_prob = tf.expand_dims(p_x.log_prob(observed), 1)
        feedback = sample if self._feedback else observed
        return (sample, log_prob), feedback


class _InfRNN(snt.AbstractModule):
    """Context, observed -> q(state) distribution."""
    def __init__(self, hparams, obs_encoder, name=None):
        super(_InfRNN, self).__init__(
            name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_encoder = obs_encoder

    def _build(self, context, observed):
        hparams = self._hparams
        enc_obs = snt.BatchApply(self._obs_encoder, n_dims=2)(observed)
        inputs = tf.concat([context, enc_obs], axis=2)
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
        return IndependentSequence(
            latent_dec.output_dist(
                snt.BatchApply(latent_dec, n_dims=2)(inputs),
                name='q_z_dist'))


class IndependentSequence(distributions.Distribution):
    """Wrapper distribution consisting of a sequence of independent events."""

    def __init__(self, item_dist, name=None):
        name = (name or self.__class__.__name__) + item_dist.name
        super(IndependentSequence, self).__init__(
            dtype=item_dist.dtype,
            reparameterization_type=item_dist.reparameterization_type,
            validate_args=item_dist.validate_args,
            allow_nan_stats=item_dist.allow_nan_stats,
            name=name)
        self._item_dist = item_dist

    def _batch_shape(self):
        return self._item_dist.batch_shape[:-1]

    def _batch_shape_tensor(self):
        return self._item_dist.batch_shape_tensor()[:-1]

    def _event_shape(self):
        return (self._item_dist.batch_shape[-1:]
                .concatenate(self._item_dist.event_shape))

    def _event_shape_tensor(self):
        return tf.concat([self._item_dist.batch_shape_tensor()[-1:],
                          self._item_dist.event_shape_tensor()], axis=0)

    def _log_prob(self, x):
        return tf.reduce_sum(self._item_dist.log_prob(x), axis=-1)

    def _prob(self, x):
        return tf.reduce_prod(self._item_dist.prob(x), axis=-1)

    def sample(self, *args, **kwargs):
        return self._item_dist.sample(*args, **kwargs)


@distributions.RegisterKL(IndependentSequence, IndependentSequence)
def _kl_independent_seq(dist_a, dist_b, name=None):
    name = name or 'KL_independent_seqs'
    with tf.name_scope(name):
        item_kl = distributions.kl(dist_a._item_dist, dist_b._item_dist)
        return tf.reduce_sum(item_kl, axis=-1)
