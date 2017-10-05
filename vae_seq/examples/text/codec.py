"""Modules for encoding and decoding observations."""

import tensorflow as tf
import sonnet as snt

from vae_seq import dist_module
from vae_seq import util


class ObsEncoder(snt.AbstractModule):
    """Integer observation -> embedding, followed bu MLP."""

    def __init__(self, hparams, name=None):
        super(ObsEncoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._embed = snt.Embed(
                vocab_size=hparams.vocab_size + hparams.oov_buckets,
                embed_dim=hparams.embed_size)
            self._mlp = util.make_mlp(
                hparams,
                hparams.obs_encoder_fc_layers)

    @property
    def output_size(self):
        """Returns the output Tensor shapes."""
        return self._mlp.output_size

    def _build(self, obs):
        return snt.Sequential([self._embed, self._mlp])(obs)


class ObsDecoder(dist_module.DistModule):
    """Inputs -> Categorical(observed; logits=mlp(inputs))"""

    def __init__(self, hparams, name=None):
        super(ObsDecoder, self).__init__(name=name)
        self._hparams = hparams

    @property
    def event_dtype(self):
        """Observations are integer IDs."""
        return tf.int32

    def dist(self, logits, name=None):
        """Constructs a Distribution from the output of the module."""
        name = name or self.module_name + "_dist"
        return tf.contrib.distributions.Categorical(
            logits=logits, dtype=tf.int32, name=name)

    def _build(self, *inputs):
        hparams = self._hparams
        num_classes = hparams.vocab_size + hparams.oov_buckets
        layers = hparams.obs_decoder_fc_hidden_layers + [num_classes]
        mlp = util.make_mlp(hparams, layers)
        logits = mlp(util.concat_features(inputs))
        return tf.reshape(logits, [-1, num_classes])
