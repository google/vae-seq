"""Evaluate a trained model."""

import tensorflow as tf
from vaeseq import util

from vaeseq.examples.midi import dataset as dataset_mod
from vaeseq.examples.midi import hparams as hparams_mod
from vaeseq.examples.midi import model

flags = tf.app.flags
flags.DEFINE_string("eval_files", None, "Location of evaluation MIDI files.")
flags.DEFINE_string("hparams", "", "HParams overrides.")
flags.DEFINE_string("log_dir", None, "Checkpoint directory.")
flags.DEFINE_integer("iters", int(1e4), "Number of iterations to average over.")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    assert FLAGS.log_dir, "Please supply a --log_dir."
    assert FLAGS.eval_files, "Please supply --eval_files."
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparams_mod.make_hparams(FLAGS.hparams)
    tf.logging.log(tf.logging.INFO, "Searching for %r.", FLAGS.eval_files)
    eval_files = tf.gfile.Glob(FLAGS.eval_files)
    assert eval_files, "No files matched by " + FLAGS.eval_files
    tf.logging.log(tf.logging.INFO, "Found %d files.", len(eval_files))
    dataset = dataset_mod.piano_roll_sequences(
        eval_files,
        util.batch_size(hparams),
        util.sequence_size(hparams),
        rate=hparams.rate)
    model.evaluate(hparams, dataset, FLAGS.log_dir, FLAGS.iters)


def entry_point():
    """Entry point for setuptools."""
    tf.app.run(main)


if __name__ == "__main__":
    entry_point()
