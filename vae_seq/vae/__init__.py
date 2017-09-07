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

"""Registry for different VAE implementations."""

from . import independent_sequence
from . import rnn
from . import srnn

VAE_TYPES = {}
VAE_TYPES["ISEQ"] = independent_sequence.IndependentSequence
VAE_TYPES["RNN"] = rnn.RNN
VAE_TYPES["SRNN"] = srnn.SRNN

def make(hparams, *args, **kwargs):
    """Create a VAE instance according to hparams.vae_type."""
    vae_type = VAE_TYPES[hparams.vae_type]
    return vae_type(hparams, *args, **kwargs)
