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

"""An agent implementation for static audio sequences."""

import numpy as np
import tensorflow as tf
from vae_seq import agent as agent_mod
from vae_seq import util


contexts_for_static_observations = agent_mod.contexts_for_static_observations

class Agent(agent_mod.EncodeObsAgent):
    """Encode the previous observation and also pass a metronome signal."""

    def __init__(self, hparams, obs_encoder, name=None):
        super(Agent, self).__init__(obs_encoder, name=name)
        self._hparams = hparams

    @property
    def context_size(self):
        return (tf.TensorShape([1]), super(Agent, self).context_size)

    @property
    def context_dtype(self):
        return (tf.float32, super(Agent, self).context_dtype)

    def context(self, agent_input, state):
        seconds = tf.to_float(agent_input) / self._hparams.audio_rate
        metronome = tf.sin(seconds * 2 * np.pi)
        return (metronome, super(Agent, self).context(agent_input, state))


def timing_input(hparams, offsets=None, dtype=tf.int32):
    """Returns agent inputs that encode per-step timing."""
    if offsets is None:
        offsets = tf.zeros([util.batch_size(hparams), 1], dtype=dtype)
    step = hparams.samples_per_step
    limit = util.sequence_size(hparams) * step
    range_ = tf.range(0, limit, step, dtype=dtype)
    # Broadcast over different offsets.
    ranges = offsets + range_
    # Expand dims to make the shape [batch_size x sequence_size x 1].
    return tf.expand_dims(ranges, 2)
