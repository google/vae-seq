# Copyright 2017 Google, Inc.,
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

"""Hyperparameters for this example."""

from vae_seq import hparams as hparams_mod

_DEFAULTS = dict(
    audio_rate=16000,
    samples_per_step=200,
    latent_size=128,
    sequence_size=40,
    obs_enc_conv_layers=[
        # layer 1
        16,  # conv output channels
        32,  # kernel width
        32,  # max-pool width
        4,   # max-pool stride
        # layer 2
        16,  # conv output channels
        32,  # kernel width
        32,  # max-pool width
        4,   # max-pool stride
    ],
    obs_dec_deconv_layers=[
        # layer 1
        16,  # input channels
        32,  # kernel width
        4,   # stride
        # layer 2
        16,  # input channels
        32,  # kernel width
        4,   # stride
    ],
)


def make_hparams(flag_value=None, **kwargs):
    """Initialize HParams with the defaults in this module."""
    init = dict(_DEFAULTS)
    init.update(kwargs)
    ret = hparams_mod.make_hparams(flag_value=flag_value, **init)
    ret.obs_shape = [ret.samples_per_step]
    return ret
