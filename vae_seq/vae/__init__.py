from . import independent_sequence
from . import rnn
from . import srnn

VAE_TYPES = {}
VAE_TYPES["ISEQ"] = independent_sequence.IndependentSequence
VAE_TYPES["RNN"] = rnn.RNN
VAE_TYPES["SRNN"] = srnn.SRNN

def make(hparams, *args, **kwargs):
    vae_type = VAE_TYPES[hparams.vae_type]
    return vae_type(hparams, *args, **kwargs)
