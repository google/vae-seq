"""Hyperparameters for this example."""

from vae_seq import hparams as hparams_mod

_DEFAULTS = dict(
    test_game_width=3,  # Number of pixels
    test_game_classes=4,  # Number of values per pixel
)


def make_hparams(flag_value=None, **kwargs):
    """Initialize HParams with the defaults in this module."""
    init = dict(_DEFAULTS)
    init.update(kwargs)
    ret = hparams_mod.make_hparams(flag_value=flag_value, **init)
    ret.obs_shape = [ret.test_game_classes]
    return ret
