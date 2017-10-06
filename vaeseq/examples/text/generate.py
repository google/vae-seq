"""Generate character sequences from a trained model."""

import tensorflow as tf

from vaeseq.examples.text import dataset as dataset_mod
from vaeseq.examples.text import hparams as hparams_mod
from vaeseq.examples.text import model

flags = tf.app.flags
flags.DEFINE_string("train_corpus", None,
                    "Location of training text, used to calculate the "
                    "vocabulary used to train the model.")
flags.DEFINE_string("hparams", "", "HParams overrides.")
flags.DEFINE_string("log_dir", None, "Checkpoint directory.")
flags.DEFINE_integer("samples", 20, "Number of generated strings")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    assert FLAGS.log_dir, "Please supply a --log_dir."
    assert FLAGS.train_corpus, "Please supply a --train_corpus."
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparams_mod.make_hparams(FLAGS.hparams)
    _unused_char_to_id, id_to_char = dataset_mod.vocabulary(
        FLAGS.train_corpus, hparams.vocab_size, hparams.oov_buckets)
    model.generate(hparams, id_to_char, FLAGS.log_dir, FLAGS.samples)


def entry_point():
    """Entry point for setuptools."""
    tf.app.run(main)


if __name__ == "__main__":
    entry_point()
