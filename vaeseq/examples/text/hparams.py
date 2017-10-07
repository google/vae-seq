"""Hyperparameters for this example."""

from vaeseq import hparams as hparams_mod

_DEFAULTS = dict(
    latent_size=16,
    sequence_size=40,
    rnn_hidden_sizes=[512, 512, 512],
    obs_encoder_fc_layers=[64, 32],
    obs_decoder_fc_hidden_layers=[64],
    embed_size=100,
    vocab_size=100,
    oov_buckets=1,
)


def make_hparams(flag_value=None, **kwargs):
    """Initialize HParams with the defaults in this module."""
    init = dict(_DEFAULTS)
    init.update(kwargs)
    ret = hparams_mod.make_hparams(flag_value=flag_value, **init)
    return ret
