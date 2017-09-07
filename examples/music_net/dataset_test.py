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

"""Tests for dataset.py functionality."""

import os
import os.path
import numpy as np
import tensorflow as tf

from examples.music_net import dataset as dataset_mod


class DatasetTest(tf.test.TestCase):

    def test_dataset_from_sequences(self):
        sequences = [tf.range(1, 10), tf.range(21, 30),]
        batch_size = 3
        obs_size = 2
        sequence_size = 4
        dataset = dataset_mod.dataset_from_sequences(
            sequences, batch_size, obs_size, sequence_size)
        iterator = tf.contrib.data.Iterator.from_dataset(dataset)
        batch = iterator.get_next()

        def _check_sequence(sequence, start, end):
            """Check that sequence was sampled correctly."""
            self.assertEqual(len(sequence), sequence_size)
            prev = start - 1
            for obs in sequence:
                self.assertEqual(len(obs), obs_size)
                for elem in obs:
                    self.assertGreater(elem, prev)
                    self.assertLess(elem, end)
                    prev = elem

        with self.test_session() as sess:
            sess.run(iterator.initializer)
            (elem1, elem2, elem3), offsets = sess.run(batch)
            _check_sequence(elem1, start=1, end=10)
            _check_sequence(elem2, start=21, end=30)
            _check_sequence(elem3, start=1, end=10)
            self.assertTrue(np.all(offsets >= 0))

    def test_musicnet_sequences(self):
        temp_path = os.path.join(tf.test.get_temp_dir(), "fake_data.npz")
        np.savez(temp_path,
                 file1=([1, 1, 2, 2], None),
                 file2=([3, 3, 4, 4, 5, 5], None),
                 file3=([6, 6, 7, 7, 8, 8, 9, 9], None))
        rate = dataset_mod.MUSICNET_SAMPLING_RATE / 2
        train_seqs = dataset_mod.load_musicnet_sequences(
            temp_path, train_frac=0.8, rate=rate, training=True)
        valid_seqs = dataset_mod.load_musicnet_sequences(
            temp_path, train_frac=0.8, rate=rate, training=False)
        with self.test_session() as sess:
            train_seqs, valid_seqs = sess.run((train_seqs, valid_seqs))
            self.assertEqual(len(train_seqs), 2)
            self.assertEqual(len(valid_seqs), 1)
            self.assertEqual(
                sum([len(seq) for seq in train_seqs + valid_seqs]), 2 + 3 + 4)

    def test_musicnet_caching(self):
        temp_path = os.path.join(tf.test.get_temp_dir(), "fake_data.npz")
        cache_path = os.path.join(tf.test.get_temp_dir(), "fake_data.cache.npz")
        np.savez(temp_path, file1=([1, 1, 2, 2], None))
        rate = dataset_mod.MUSICNET_SAMPLING_RATE / 2
        _unused_seqs = dataset_mod.load_musicnet_sequences(
            temp_path, train_frac=1.0, rate=rate, cache_path=cache_path)
        self.assertAllClose(np.load(cache_path)["file1"][0], [1.25, 1.75])


if __name__ == "__main__":
    tf.test.main()
