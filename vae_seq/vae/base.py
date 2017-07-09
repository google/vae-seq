import abc
import six
import tensorflow as tf
import sonnet as snt

@six.add_metaclass(abc.ABCMeta)
class VAEBase(snt.AbstractModule):
    """Base class for Sequential VAE implementations."""

    def __init__(self, hparams, obs_encoder, obs_decoder, name=None):
        super(VAEBase, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._obs_encoder = obs_encoder
        self._obs_decoder = obs_decoder

    def _build(self, context=None, observed=None):
        """Constructs the VAE.

        Args:
          context: An optional [batch_size x sequence_size x K] tensor of per-step
            context features.
          observed: A [batch_size x sequence_size] x observation shape tensor
            of observations, used for training and evaluation.
        """
        hparams = self._hparams
        if context is None:
            context = tf.zeros(
                [hparams.batch_size, hparams.sequence_size, 0],
                name='no_context')
        if observed is None:
            observed = tf.zeros(
                [hparams.batch_size, hparams.sequence_size] + hparams.obs_shape,
                name='dummy_observed')
            enc_observed = tf.zeros(
                [hparams.batch_size, hparams.sequence_size,
                 hparams.enc_obs_size],
                name='dummy_enc_observed')
        else:
            enc_observed = snt.BatchApply(self._obs_encoder, n_dims=2)(observed)
        return self._build_vae(context, observed, enc_observed)

    @abc.abstractmethod
    def _build_vae(self, context, observed, enc_observed):
        return
