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

"""We can view an RNN as a VAE with no latent variables:

Notation:
 - d_1:T are the (deterministic) RNN outputs.
 - x_1:T are the observed states.
 - c_1:T are per-timestep inputs.

        Generative model
      =====================
      x_1               x_t
       ^                 ^
       |                 |
      d_1 ------------> d_t
       ^                 ^
       |                 |
      c_1               c_t
"""

import tensorflow as tf
from tensorflow.contrib import distributions

from .. import util
from .. import vae_module

class RNN(vae_module.VAECore):
    """Implementation of an RNN as a sequential VAE where all latent
       variables are deterministic."""

    def __init__(self, hparams, obs_encoder, obs_decoder, name=None):
        super(RNN, self).__init__(hparams, obs_encoder, obs_decoder, name)
        with self._enter_variable_scope():
            self._d_core = util.make_rnn(hparams, name="d_core")

    @property
    def state_size(self):
        return self._d_core.state_size

    def _next_state(self, d_state, event=None):
        del event  # Not used.
        return d_state

    def _initial_state(self, batch_size):
        return self._d_core.initial_state(batch_size)

    def _build(self, input_, d_state):
        d_out, d_state = self._d_core(util.concat_features(input_), d_state)
        return self._obs_decoder(d_out), d_state

    def _infer_latents(self, inputs, observed):
        """Because the RNN latent state is fully deterministic, there's no
           need to do two passes over the training data."""
        del inputs  # Not used.
        batch_size = util.batch_size_from_nested_tensors(observed)
        sequence_size = util.sequence_size_from_nested_tensors(observed)
        divs = tf.zeros([batch_size, sequence_size], name="divergences")
        return None, divs
