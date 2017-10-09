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
        self.hparams = hparams_mod.make_hparams(rnn_hidden_sizes=[4, 4],
                                                check_numerics=True)
        self.model = model_mod.Model(self.hparams, session_params)


if __name__ == "__main__":
    tf.test.main()
