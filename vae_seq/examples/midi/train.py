"""Train a model over MIDI music."""

import tensorflow as tf
from vae_seq import util

from vae_seq.examples.midi import dataset as dataset_mod
from vae_seq.examples.midi import hparams as hparams_mod
from vae_seq.examples.midi import model

flags = tf.app.flags
flags.DEFINE_string("train_files", None, "Location of training MIDI files.")
flags.DEFINE_string("valid_files", None, "Location of validation MIDI files.")
flags.DEFINE_string("log_dir", None, "Checkpoint directory.")
flags.DEFINE_string("hparams", "", "HParams overrides.")
flags.DEFINE_integer("iters", int(1e6), "Number of training iterations")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    assert FLAGS.log_dir, "Please supply a --log_dir."
    assert FLAGS.train_files, "Please supply --train_files."
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparams_mod.make_hparams(FLAGS.hparams)
    tf.logging.log(
        tf.logging.INFO,
        "Searching for training files %r.", FLAGS.train_files)
    train_files = tf.gfile.Glob(FLAGS.train_files)
    assert train_files, "No files matched by " + FLAGS.train_files
    tf.logging.log(tf.logging.INFO, "Found %d files.", len(train_files))
    dataset = dataset_mod.piano_roll_sequences(
        train_files,
        util.batch_size(hparams),
        util.sequence_size(hparams),
        rate=hparams.rate)

    if FLAGS.valid_files:
        tf.logging.log(
            tf.logging.INFO,
            "Searching for validation files %r.", FLAGS.train_files)
        valid_files = tf.gfile.Glob(FLAGS.valid_files)
        assert valid_files, "No files matched by " + FLAGS.valid_files
        tf.logging.log(tf.logging.INFO, "Found %d files.", len(valid_files))
        valid_dataset = dataset_mod.piano_roll_sequences(
            valid_files,
            util.batch_size(hparams),
            util.sequence_size(hparams),
            rate=hparams.rate)

    model.train(hparams, dataset,
                FLAGS.log_dir, FLAGS.iters, valid_dataset=valid_dataset)


def entry_point():
    """Entry point for setuptools."""
    tf.app.run(main)


if __name__ == "__main__":
    entry_point()
