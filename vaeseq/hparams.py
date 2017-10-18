"""Hyperparameters used in this library."""

import tensorflow as tf

_DEFAULTS = dict(
    # Number of latent units per time step
    latent_size=4,

    # Model parameters
    obs_encoder_fc_layers=[256, 128],
    obs_decoder_fc_hidden_layers=[256],
    latent_decoder_fc_layers=[256],
    rnn_hidden_sizes=[32],

    # Default activation (relu/elu/etc.)
    activation='relu',

    # Postivitity constraint (softplus/exp/etc.)
    positive_projection='softplus',

    # VAE params
    divergence_strength_start=1e-5,  # scale on divergence penalty.
    divergence_strength_half=1e5,  # in global-steps.
    vae_type='SRNN',  # see vae.VAE_TYPES.
    use_monte_carlo_kl=False,
    srnn_use_res_q=True,
    learning_rate=0.0001,

    # Agent params
    reinforce_agent_across_timesteps=True,
    train_agent_from_model=False,
    agent_learning_rate=0.0001,

    # Training parameters
    batch_size=20,
    sequence_size=5,
    clip_gradient_norm=1.,
    check_numerics=True,
)

def make_hparams(flag_value=None, **kwargs):
    """Initialize HParams with the defaults in this module."""
    init = dict(_DEFAULTS)
    init.update(kwargs)
    ret = tf.contrib.training.HParams(**init)
    if flag_value:
        ret.parse(flag_value)
    return ret
