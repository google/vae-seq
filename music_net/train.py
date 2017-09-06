"""Train a model over the MusicNet dataset:

John Thickstun, Zaid Harchaoui, Sham Kakade.
Learning Features of Music from Scratch.
https://arxiv.org/abs/1611.09827
"""

import os.path
import tensorflow as tf
from vae_seq import util

from music_net import dataset as dataset_mod
from music_net import hparams as hparams_mod
from music_net import model

flags = tf.app.flags
flags.DEFINE_string(
    "musicnet_file", None,
    "Location of the downloaded musicnet.npz. "
    "See: http://homes.cs.washington.edu/~thickstn/musicnet.html")
flags.DEFINE_bool(
    "use_cache", True,
    "Cache the resampled training data.")
flags.DEFINE_string(
    "cache_file", None,
    "Path to cache file for resampled data. If not specified, a "
    "file adjacent to --musicnet_file will be used.")
flags.DEFINE_string("log_dir", None, "Checkpoint directory.")
flags.DEFINE_string("hparams", "", "HParams overrides.")
flags.DEFINE_integer("iters", int(1e6), "Number of training iterations")
flags.DEFINE_float("train_frac", 0.01, "Fraction of musicnet to train on.")

FLAGS = flags.FLAGS


def cache_file(hparams):
    """Determines which cache file to use, if any."""
    if not FLAGS.use_cache:
        return None
    if FLAGS.cache_file:
        return FLAGS.cache_file
    dirname = os.path.dirname(FLAGS.musicnet_file)
    fname = "musicnet.train.frac_%g.rate_%d.npz" % (FLAGS.train_frac,
                                                    hparams.audio_rate)
    return os.path.join(dirname, fname)


def main(argv):
    del argv
    assert FLAGS.log_dir, "Please supply a --log_dir."
    assert FLAGS.musicnet_file, "Please supply a --musicnet_file."
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparams_mod.make_hparams(FLAGS.hparams)
    tf.logging.log(tf.logging.INFO, "Loading sequences.")
    train_sequences = dataset_mod.load_musicnet_sequences(
        FLAGS.musicnet_file, FLAGS.train_frac,
        rate=hparams.audio_rate,
        cache_path=cache_file(hparams),
        training=True)
    tf.logging.log(tf.logging.INFO, "Finished loading sequences.")
    dataset = dataset_mod.dataset_from_sequences(
        train_sequences, util.batch_size(hparams),
        hparams.samples_per_step, util.sequence_size(hparams))
    model.train(hparams, dataset, FLAGS.log_dir, FLAGS.iters)


if __name__ == "__main__":
    tf.app.run(main)
