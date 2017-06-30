from . import independent_sequence
from . import srnn

VAE_TYPES = {}
VAE_TYPES['IndependentSequence'] = independent_sequence.IndependentSequenceVAE
VAE_TYPES['SRNN'] = srnn.SRNN
