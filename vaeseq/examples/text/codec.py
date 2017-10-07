"""Modules for encoding and decoding observations."""

import tensorflow as tf
import sonnet as snt

from vaeseq import codec
from vaeseq import util


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


class ObsDecoder(codec.CategoricalMLPObsDecoder):
    """Inputs -> Categorical(observed; logits=mlp(inputs))"""

    def __init__(self, hparams, name=None):
        num_classes = hparams.vocab_size + hparams.oov_buckets
        super(ObsDecoder, self).__init__(hparams, num_classes, name=name)
