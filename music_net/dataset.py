"""Dataset for iterating over Sampled music data."""

import functools
import numpy as np
import scipy.signal
import tensorflow as tf

MUSICNET_SAMPLING_RATE = 44100


def _concat_tensors(tensors):
    """Returns a Dataset that returns a list of Tensors."""
    dataset = None
    for tensor in tensors:
        tensor_ds = tf.contrib.data.Dataset.from_tensors(tensor)
        if dataset is None:
            dataset = tensor_ds
        else:
            dataset = dataset.concatenate(tensor_ds)
    return dataset


def _sample_sequence(obs_size, sequence_size, samples):
    """Samples a sequence into sequences of observations."""
    samples_length = tf.shape(samples)[0]
    total_obs = samples_length // obs_size
    take_obs = tf.minimum(sequence_size, total_obs)
    slice_size = take_obs * obs_size
    slice_start = tf.random_uniform(
        shape=[1],
        minval=0,
        maxval=1 + samples_length - slice_size,
        dtype=tf.int32)
    slice_ = tf.pad(
        tf.slice(samples, slice_start, [slice_size]),
        [[0, sequence_size * obs_size - slice_size]])
    ret = tf.reshape(slice_, [sequence_size, obs_size])
    return tf.contrib.data.Dataset.from_tensors((ret, slice_start))


def dataset_from_sequences(sequences, batch_size, obs_size, sequence_size):
    """Returns a Dataset for reading batches of samples.

    Args:
      sequences: a list of Tensors containing samples.
      batch_size: the batch dimension.
      obs_size: the number of samples per observation.
      sequence_size: the number of observations per batch.

    Returns:
      A tf.contrib.data.Dataset.

    The returned Dataset produces two tensors:
      samples: a [batch_size x sequence_size x obs_size] float32 Tensor.
      offsets: a [batch_size] int32 Tensor of offsets into the sequence.
    """
    return (_concat_tensors(sequences)
            .repeat()
            .flat_map(
                functools.partial(_sample_sequence, obs_size, sequence_size))
            .batch(tf.to_int64(batch_size)))


def load_musicnet_sequences(
        musicnet_file, train_frac, rate=MUSICNET_SAMPLING_RATE,
        training=True, cache_path=None):
    """Loads audio sequences from the MusicNet dataset.

    Args:
      musicnet_file: the musicnet.npz.
      train_frac: the proportion of the files to use for training.
      rate: the desired sampling rate.
      training: whether to return the training portion or the rest.
      cache_path: path to check/save cached resampled sequences.

    Returns:
      A list of 1-D sample Tensors.
    """
    loaded_from_cache = False
    if cache_path is not None:
        try:
            data = np.load(cache_path)
            files = [fname for fname in data.files]
            loaded_from_cache = True
        except IOError, err:
            tf.logging.log(
                tf.logging.INFO,
                "Couldn't load cache {}: {}".format(cache_path, err))
    if not loaded_from_cache:
        # TODO: the gFile method is much slower since there's no mmapping.
        # with tf.gfile.Open(musicnet_file, mode="rb") as data_file:
        #     data = np.load(data_file)
        data = np.load(musicnet_file)
        files = [fname for fname in data.files]
        files.sort(key=hash)
        cutoff = int(len(files) * train_frac)
        files = files[:cutoff] if training else files[cutoff:]
    sequences = [data[fname][0] for fname in files]
    if not loaded_from_cache and rate != MUSICNET_SAMPLING_RATE:
        downsample = rate / float(MUSICNET_SAMPLING_RATE)
        sequences = [
            scipy.signal.resample(samples, int(len(samples) * downsample))
            for samples in sequences]
        if cache_path is not None:
            try:
                itrees = [data[fname][1] for fname in files]
                cache_contents = {
                    fname: (samples, itree)
                    for fname, samples, itree in zip(files, sequences, itrees)
                }
                np.savez(cache_path, **cache_contents)
            except IOError, err:
                tf.logging.log(
                    tf.logging.WARN,
                    "Couldn't save cache {}: {}".format(cache_path, err))
    return [tf.constant(samples, dtype=tf.float32) for samples in sequences]
