"""Tests for dataset.py functionality."""

import os.path
import numpy as np
import pretty_midi
import tensorflow as tf

from vaeseq.examples.midi import dataset as dataset_mod


def _write_midi(note):
    """Write a temporary MIDI file with a note playing for one second."""
    temp_path = os.path.join(tf.test.get_temp_dir(), "note_{}.mid".format(note))
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(127, note, 0.0, 1.0))
    midi.instruments.append(instrument)
    midi.write(temp_path)
    return temp_path


class DatasetTest(tf.test.TestCase):

    def test_piano_roll_sequences(self):
        filenames = [_write_midi(5), _write_midi(7)]
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


if __name__ == "__main__":
    tf.test.main()
