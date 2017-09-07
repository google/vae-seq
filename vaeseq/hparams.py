# Copyright 2018 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    positive_eps=1e-5,

    # VAE params
    divergence_strength_start=1e-5,  # scale on divergence penalty.
    divergence_strength_half=1e5,  # in global-steps.
    vae_type='SRNN',  # see vae.VAE_TYPES.
    use_monte_carlo_kl=False,
    srnn_use_res_q=True,

    # Training parameters
    learning_rate=0.0001,
    l1_regularization=0.0,
    l2_regularization=0.01,
    batch_size=32,
    sequence_size=5,
    clip_gradient_norm=1.,
    check_numerics=True,

    # Evaluation parameters
    log_prob_samples=10,  # number of latent samples to average over.
)

def make_hparams(flag_value=None, **kwargs):
    """Initialize HParams with the defaults in this module."""
    init = dict(_DEFAULTS)
    init.update(kwargs)
    ret = tf.contrib.training.HParams(**init)
    if flag_value:
        ret.parse(flag_value)
    return ret
