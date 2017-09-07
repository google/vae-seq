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
        return tf.data.Dataset.from_tensor_slices(sequences)

    batch_size = tf.to_int64(batch_size)
    return (tf.data.Dataset.from_tensor_slices(filenames)
            .interleave(_to_piano_roll_dataset,
                        cycle_length=batch_size * 5,
                        block_length=1)
            .repeat()
            .shuffle(1000)
            .batch(batch_size))


def piano_roll_to_midi(piano_roll, sample_rate):
    """Convert the piano roll to a PrettyMIDI object.
    See: http://github.com/craffel/examples/reverse_pianoroll.py
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    midi.instruments.append(instrument)
    padded_roll = np.pad(piano_roll, [(1, 1), (0, 0)], mode='constant')
    changes = np.diff(padded_roll, axis=0)
    notes = np.full(piano_roll.shape[1], -1, dtype=np.int)
    for tick, pitch in zip(*np.where(changes)):
        prev = notes[pitch]
        if prev == -1:
            notes[pitch] = tick
            continue
        notes[pitch] = -1
        instrument.notes.append(pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=prev / float(sample_rate),
            end=tick / float(sample_rate)))
    return midi


def write_test_note(path, duration, note):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(100, note, 0.0, duration))
    midi.instruments.append(instrument)
    midi.write(path)
