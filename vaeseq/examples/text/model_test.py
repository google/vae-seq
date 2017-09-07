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

import io
import os.path
import tensorflow as tf
from vaeseq import model_test

from vaeseq.examples.text import hparams as hparams_mod
from vaeseq.examples.text import model as model_mod


class ModelTest(model_test.ModelTest):

    def _write_corpus(self, text):
        """Writes the given text to a temporary file and returns the path."""
        temp_path = os.path.join(self.get_temp_dir(), "corpus.txt")
        with io.open(temp_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(tf.compat.as_text(text))
        return temp_path

    def _setup_model(self, session_params):
        self.train_dataset = self._write_corpus("1234567890" * 100)
        self.valid_dataset = self._write_corpus("123" * 20)
        self.hparams = hparams_mod.make_hparams(
            vocab_size=5,
            rnn_hidden_sizes=[4, 4],
            obs_encoder_fc_layers=[32, 16],
            obs_decoder_fc_hidden_layers=[32],
            latent_decoder_fc_layers=[32],
            check_numerics=True)
        vocab_corpus = self.train_dataset
        self.model = model_mod.Model(self.hparams, session_params, vocab_corpus)


if __name__ == "__main__":
    tf.test.main()
