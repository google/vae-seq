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

"""Tests for dataset.py functionality."""

import os.path
import numpy as np
import pretty_midi
import tensorflow as tf

from vaeseq.examples.midi import dataset as dataset_mod


class DatasetTest(tf.test.TestCase):

    def _write_midi(self, note):
        """Write a temporary MIDI file with a note playing for one second."""
        temp_path = os.path.join(self.get_temp_dir(),
                                 "note_{}.mid".format(note))
        dataset_mod.write_test_note(temp_path, 1.0, note)
        return temp_path

    def test_piano_roll_sequences(self):
        filenames = [self._write_midi(5), self._write_midi(7)]
        batch_size = 2
        sequence_size = 3
        rate = 2
        dataset = dataset_mod.piano_roll_sequences(
            filenames, batch_size, sequence_size, rate)
        iterator = dataset.make_initializable_iterator()
        batch = iterator.get_next()
        with self.test_session() as sess:
            sess.run(iterator.initializer)
            batch = sess.run(batch)
            self.assertAllEqual(batch.shape, [batch_size, sequence_size, 128])
            batch_idx, time_idx, note_idx = np.where(batch)
            self.assertAllEqual(batch_idx, [0, 0, 1, 1])
            self.assertAllEqual(time_idx, [0, 1, 0, 1])
            self.assertEqual(note_idx[0], note_idx[1])
            self.assertIn(note_idx[0], (5, 7))
            self.assertEqual(note_idx[2], note_idx[3])
            self.assertIn(note_idx[1], (5, 7))

    def test_piano_roll_to_midi(self):
        np.random.seed(0)
        piano_roll = np.random.uniform(size=(200, 128)) > 0.5
        midi = dataset_mod.piano_roll_to_midi(piano_roll, 2)
        self.assertAllEqual(piano_roll.T, midi.get_piano_roll(2) > 0)


if __name__ == "__main__":
    tf.test.main()
