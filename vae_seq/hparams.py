import tensorflow as tf

def _defaults():
    # Shape of observed events, set in game-specific code.
    obs_shape = None

    # Size of encoded (flat) observations
    enc_obs_size = 128

    # Number of latent units per time step
    latent_size = 4

    # Model parameters
    obs_encoder_fc_layers = [256]
    obs_decoder_fc_layers = [256]
    latent_decoder_fc_layers = [256]
    rnn_hidden_sizes = [32]

    # Default activation (relu/elu/etc.)
    activation = 'relu'

    # Postivitity constraint (softplus/exp/etc.)
    positive_projection = 'softplus'

    # VAE params
    divergence_strength_halfway_point = 1e4  # global steps.
    vae_type = 'SRNN'  # See vae.VAE_TYPES
    use_monte_carlo_kl = False
    srnn_use_res_q = True

    # Training parameters
    batch_size = 20
    sequence_size = 5
    learning_rate = 0.0001
    check_numerics = False

    return locals()


def HParams(**kwargs):
    init = _defaults()
    init.update(kwargs)
    return tf.contrib.training.HParams(**init)
