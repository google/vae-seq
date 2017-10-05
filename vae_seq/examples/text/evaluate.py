"""Evaluate a trained model."""

import tensorflow as tf
from vae_seq import util

from vae_seq.examples.text import dataset as dataset_mod
from vae_seq.examples.text import hparams as hparams_mod
from vae_seq.examples.text import model

flags = tf.app.flags
flags.DEFINE_string("train_corpus", None,
                    "Location of training text, used to calculate the "
                    "vocabulary used to train the model.")
flags.DEFINE_string("eval_corpus", None, "Location of evaluation text.")
flags.DEFINE_string("hparams", "", "HParams overrides.")
flags.DEFINE_string("log_dir", None, "Checkpoint directory.")
flags.DEFINE_integer("iters", int(1e4), "Number of iterations to average over.")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    assert FLAGS.log_dir, "Please supply a --log_dir."
    assert FLAGS.train_corpus, "Please supply a --train_corpus."
    assert FLAGS.eval_corpus, "Please supply a --eval_corpus."
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparams_mod.make_hparams(FLAGS.hparams)
    char_to_id, _unused_id_to_char = dataset_mod.vocabulary(
        FLAGS.train_corpus, hparams.vocab_size, hparams.oov_buckets)
    dataset = dataset_mod.characters(FLAGS.train_corpus,
                                     util.batch_size(hparams),
                                     util.sequence_size(hparams))
    model.evaluate(hparams, dataset, char_to_id, FLAGS.log_dir, FLAGS.iters)


def entry_point():
    """Entry point for setuptools."""
    tf.app.run(main)


if __name__ == "__main__":
    entry_point()
