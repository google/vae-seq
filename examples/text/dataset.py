"""Dataset for iterating over text."""

import collections
import numpy as np
import tensorflow as tf


def _split_string(string):
    """Splits a byte string into an array of character bytes."""
    text = tf.compat.as_text(string)
    ret = np.empty(len(text), dtype=np.object)
    for i, char in enumerate(text):
        ret[i] = tf.compat.as_bytes(char)
    return ret


def vocabulary(filename, max_size=None, num_oov_buckets=1):
    """Builds vocabulary and ID lookup tables from the given file."""

    def _unique_chars(filename):
        """Returns the used alphabet as an array of strings."""
        counts = collections.Counter()
        with tf.gfile.Open(filename) as file_:
            for line in file_:
                counts.update(_split_string(line))
        alphabet = [k for (k, _) in counts.most_common(max_size)]
        alphabet.sort()
        return np.asarray(alphabet, dtype=np.object)

    chars, = tf.py_func(_unique_chars, [filename], [tf.string])
    char_to_id = tf.contrib.lookup.index_table_from_tensor(
        chars, num_oov_buckets=num_oov_buckets)
    id_to_char = tf.contrib.lookup.index_to_string_table_from_tensor(chars, " ")
    return char_to_id, id_to_char


def characters(filename, batch_size, sequence_size):
    """Returns a dataset of characters from the given file."""

    def _to_chars(line):
        """string scalar -> Dataset of characters (string scalars)."""
        chars, = tf.py_func(_split_string, [line + "\n"], [tf.string])
        chars.set_shape([None])
        return tf.contrib.data.Dataset.from_tensor_slices(chars)

    return (tf.contrib.data.TextLineDataset([filename])
            .flat_map(_to_chars)
            .repeat()
            .batch(tf.to_int64(sequence_size))
            .shuffle(1000)
            .batch(tf.to_int64(batch_size)))
