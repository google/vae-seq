"""Generate character sequences from a trained model."""

import tensorflow as tf

from vaeseq.examples.midi import hparams as hparams_mod
from vaeseq.examples.midi import model

flags = tf.app.flags
flags.DEFINE_string("hparams", "", "HParams overrides.")
flags.DEFINE_string("log_dir", None, "Checkpoint directory.")
flags.DEFINE_string("out_dir", None, "Where to place output files.")
flags.DEFINE_integer("samples", 20, "Number of generated strings")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    assert FLAGS.log_dir, "Please supply a --log_dir."
    assert FLAGS.out_dir, "Please supply an --out_dir."
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparams_mod.make_hparams(FLAGS.hparams)
    model.generate(hparams, FLAGS.log_dir, FLAGS.out_dir, FLAGS.samples)


def entry_point():
    """Entry point for setuptools."""
    tf.app.run(main)


if __name__ == "__main__":
    entry_point()
