"""Modules for encoding and decoding observations."""

import tensorflow as tf

from vaeseq import batch_dist
from vaeseq import codec
from vaeseq import util


ObsEncoder = codec.MLPObsEncoder


class ObsDecoder(codec.MLPObsDecoderBase):
    """Inputs -> independent Bernoulli activations."""

    def __init__(self, hparams, name=None):
        super(ObsDecoder, self).__init__(hparams, 128, name=name)

    @property
    def event_dtype(self):
        return tf.bool

    @property
    def event_size(self):
        return tf.TensorShape([128])

    def dist(self, params, name=None):
        """Constructs the output Distribution."""
        name_prefix = name or self.module_name
        note_dist = tf.distributions.Bernoulli(
            logits=params,
            dtype=self.event_dtype,
            name=name_prefix + "_note_dist")
        return batch_dist.BatchDistribution(
            note_dist,
            ndims=1,
            name=name_prefix+"_scale_dist")
