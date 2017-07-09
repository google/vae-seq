import tensorflow as tf

from vae_seq import hparams as hparams_mod
from vae_seq import obs_layers
from vae_seq import util
from vae_seq import vae as vae_mod

def _BuildVAE(vae_type):
    hparams = hparams_mod.HParams(obs_shape=[1], vae_type=vae_type)
    obs_encoder = obs_layers.ObsEncoder(hparams)
    obs_decoder = obs_layers.ObsDecoder(hparams)
    return vae_mod.VAE_TYPES[vae_type](hparams, obs_encoder, obs_decoder)

class VAETest(tf.test.TestCase):
    def testIndependentSequenceConstruction(self):
        vae = _BuildVAE('IndependentSequence')
        self.assertTrue(isinstance(vae(), util.VAETensors))

    def testSRNNConstruction(self):
        vae = _BuildVAE('SRNN')
        self.assertTrue(isinstance(vae(), util.VAETensors))


if __name__ == '__main__':
    tf.test.main()
