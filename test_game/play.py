"""Play the test game using a trained model."""

import tensorflow as tf
from test_game import model

flags = tf.app.flags
flags.DEFINE_string("log_dir", None, "Checkpoint directory.")
flags.DEFINE_string("hparams", "", "HParams overrides.")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    assert FLAGS.log_dir, "Please supply a --log_dir."
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = model.hparams(FLAGS.hparams)
    model.play(hparams, FLAGS.log_dir)


if __name__ == "__main__":
    tf.app.run(main)
