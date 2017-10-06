"""Dataset for iterating over MIDI files."""

from __future__ import print_function

import numpy as np
import pretty_midi
import tensorflow as tf


def piano_roll_sequences(filenames, batch_size, sequence_size, rate=100):
    """Returns a dataset of piano roll sequences from the given files.."""

    def _to_piano_roll(filename, sequence_size):
        """Load a file and return consecutive piano roll sequences."""
        try:
            midi = pretty_midi.PrettyMIDI(tf.compat.as_text(filename))
        except Exception:
            print("Skipping corrupt MIDI file", filename)
            return np.zeros([0, sequence_size, 128], dtype=np.bool)
        roll = np.asarray(midi.get_piano_roll(rate).transpose(), dtype=np.bool)
        assert roll.shape[1] == 128
        # Pad the roll to a multiple of sequence_size
        length = len(roll)
        remainder = length % sequence_size
        if remainder:
            new_length = length + sequence_size - remainder
            roll = np.resize(roll, (new_length, 128))
            roll[length:, :] = False
            length = new_length
        return np.reshape(roll, (length // sequence_size, sequence_size, 128))

    def _to_piano_roll_dataset(filename):
        """Filename (string scalar) -> Dataset of piano roll sequences."""
        sequences, = tf.py_func(_to_piano_roll,
                                [filename, sequence_size],
                                [tf.bool])
        sequences.set_shape([None, None, 128])
        return tf.contrib.data.Dataset.from_tensor_slices(sequences)

    batch_size = tf.to_int64(batch_size)
    return (tf.contrib.data.Dataset.from_tensor_slices(filenames)
            .interleave(_to_piano_roll_dataset,
                        cycle_length=batch_size * 5,
                        block_length=1)
            .repeat()
            .shuffle(1000)
            .batch(batch_size))
