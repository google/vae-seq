"""Module parameterizing VAE latent variables."""

import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions

from .. import util

class LatentDecoder(snt.AbstractModule):
    """Inputs -> P(latent | inputs)"""

    def __init__(self, hparams, name=None):
        super(LatentDecoder, self).__init__(name=name)
        self._hparams = hparams

    def _build(self, *inputs):
        hparams = self._hparams
        mlp = util.make_mlp(
            hparams,
            hparams.latent_decoder_fc_layers + [hparams.latent_size * 2])
        dist_params = mlp(util.concat_features(inputs))
        loc = dist_params[:, :hparams.latent_size]
        scale = util.positive_projection(hparams)(
            dist_params[:, hparams.latent_size:])
        return (loc, scale)

    @staticmethod
    def output_dist((loc, scale_diag), name=None):
        """Constructs a Distrubution from the output of the module."""
        return distributions.MultivariateNormalDiag(loc, scale_diag, name=name)

    def dist(self, *inputs):
        """Returns p(latent | inputs)."""
        return self.output_dist(self(*inputs), name=self.module_name + "Dist")

    @property
    def event_dtype(self):
        """The data type of the latent variables."""
        return tf.float32
