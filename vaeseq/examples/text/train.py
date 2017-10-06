"""Train a model over text data, character-by-character."""

import tensorflow as tf
from vaeseq import util

from vaeseq.examples.text import dataset as dataset_mod
from vaeseq.examples.text import hparams as hparams_mod
from vaeseq.examples.text import model

flags = tf.app.flags
flags.DEFINE_string("train_corpus", None, "Location of training text.")
flags.DEFINE_string("valid_corpus", None, "Location of validation text.")
flags.DEFINE_string("log_dir", None, "Checkpoint directory.")
flags.DEFINE_string("hparams", "", "HParams overrides.")
flags.DEFINE_integer("iters", int(1e6), "Number of training iterations")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    assert FLAGS.log_dir, "Please supply a --log_dir."
    assert FLAGS.train_corpus, "Please supply a --train_corpus."
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparams_mod.make_hparams(FLAGS.hparams)
    char_to_id, id_to_char = dataset_mod.vocabulary(
        FLAGS.train_corpus, hparams.vocab_size)
    batch_size = util.batch_size(hparams)
    sequence_size = util.sequence_size(hparams)
    dataset = dataset_mod.characters(FLAGS.train_corpus,
                                     batch_size,
                                     sequence_size)
    valid_dataset = None
    if FLAGS.valid_corpus is not None:
        valid_dataset = dataset_mod.characters(FLAGS.valid_corpus,
                                               batch_size,
                                               sequence_size)
    model.train(hparams, dataset, char_to_id, id_to_char,
                FLAGS.log_dir, FLAGS.iters, valid_dataset=valid_dataset)


def entry_point():
    """Entry point for setuptools."""
    tf.app.run(main)


if __name__ == "__main__":
    entry_point()
