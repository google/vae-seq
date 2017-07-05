import tensorflow as tf

from vae_seq import hparams as hparams_mod
from vae_seq import obs_layers
from vae_seq.vae import independent_sequence as iseq


class IndependentSequenceVAETest(tf.test.TestCase):

    def _build_vae(self):
        hparams = hparams_mod.HParams(obs_shape=[1])
        obs_encoder = obs_layers.ObsEncoder(hparams)
        obs_decoder = obs_layers.ObsDecoder(hparams)
        iseq.IndependentSequenceVAE(hparams, obs_encoder, obs_decoder)

    def testConstruction(self):
        self._build_vae()


if __name__ == '__main__':
    tf.test.main()
    
