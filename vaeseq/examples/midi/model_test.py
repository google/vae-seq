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

"""Tests for training and generating graphs."""

import os.path
import tensorflow as tf
from vaeseq import model_test

from vaeseq.examples.midi import dataset as dataset_mod
from vaeseq.examples.midi import hparams as hparams_mod
from vaeseq.examples.midi import model as model_mod


class ModelTest(model_test.ModelTest):

    def _write_midi(self, note):
        """Write a temporary MIDI file with a note playing for one second."""
        temp_path = os.path.join(self.get_temp_dir(),
                                 "note_{}.mid".format(note))
        dataset_mod.write_test_note(temp_path, 1.0, note)
        return temp_path

    def _setup_model(self, session_params):
        self.train_dataset = [self._write_midi(5), self._write_midi(7)]
        self.valid_dataset = [self._write_midi(5), self._write_midi(6)]
        self.hparams = hparams_mod.make_hparams(
            rnn_hidden_sizes=[4, 4],
            obs_encoder_fc_layers=[32, 16],
            obs_decoder_fc_hidden_layers=[32],
            latent_decoder_fc_layers=[32],
            check_numerics=True)
        self.model = model_mod.Model(self.hparams, session_params)


if __name__ == "__main__":
    tf.test.main()
