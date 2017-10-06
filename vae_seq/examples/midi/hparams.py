"""Hyperparameters for this example."""

from vae_seq import hparams as hparams_mod

_DEFAULTS = dict(
    rnn_hidden_sizes=[32, 32],
    obs_encoder_fc_layers=[64, 32],
    obs_decoder_fc_hidden_layers=[64],
    latent_size=16,
    sequence_size=64,
    rate=32,
)


def make_hparams(flag_value=None, **kwargs):
    """Initialize HParams with the defaults in this module."""
    init = dict(_DEFAULTS)
    init.update(kwargs)
    ret = hparams_mod.make_hparams(flag_value=flag_value, **init)
    # Observations are individual note levels, 128 of them per tick.
    ret.obs_shape = [128]
    return ret
