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

"""Hyperparameters for this example."""

from vaeseq import hparams as hparams_mod

_DEFAULTS = dict(
    latent_size=16,
    sequence_size=40,
    obs_encoder_fc_layers=[64, 64],
    obs_decoder_fc_hidden_layers=[64],
    latent_decoder_fc_layers=[64],
    rnn_hidden_sizes=[64],
    game="CartPole-v0",
    game_output_size=[4],
    game_action_space=2,
    batch_size=32,
    explore_temp=0.5,
    l2_regularization=0.1,
)


def make_hparams(flag_value=None, **kwargs):
    """Initialize HParams with the defaults in this module."""
    init = dict(_DEFAULTS)
    init.update(kwargs)
    ret = hparams_mod.make_hparams(flag_value=flag_value, **init)
    return ret
