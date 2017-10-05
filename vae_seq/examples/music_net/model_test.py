"""Tests for training and generating graphs."""

import tensorflow as tf
from vae_seq import util

from vae_seq.examples.music_net import dataset as dataset_mod
from vae_seq.examples.music_net import hparams as hparams_mod
from vae_seq.examples.music_net import model


def _hparams(vae_type):
    """HParams used for testing the given VAE type."""
    hparams = hparams_mod.make_hparams()
    hparams.vae_type = vae_type
    hparams.check_numerics = True
    return hparams


def _dataset(hparams):
    """Mock dataset used to test trainign."""
    sequence_size = util.sequence_size(hparams)
    return dataset_mod.dataset_from_sequences(
        [tf.zeros([sequence_size * 2])],
        util.batch_size(hparams),
        hparams.samples_per_step,
        sequence_size)


class ModelTest(tf.test.TestCase):

    def _test_training(self, vae_type):
        """Test the training graph for the given VAE type."""
        hparams = _hparams(vae_type)
        vae = model.make_vae(hparams)
        dataset = _dataset(hparams)
        iterator = dataset.make_initializable_iterator()
        observed, offsets = iterator.get_next()
        train_op, debug_tensors = model.train_graph(
            hparams, vae, observed, offsets)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            debug_vals = sess.run(debug_tensors)
            self.assertGreaterEqual(debug_vals["divergence"], 0)
            for _ in range(100):
                sess.run(train_op)
            debug_vals2 = sess.run(debug_tensors)
            self.assertGreater(debug_vals2["elbo_opt"], debug_vals["elbo_opt"])

    def _test_generating(self, vae_type):
        """Test the generation graph for the given VAE type."""
        hparams = _hparams(vae_type)
        vae = model.make_vae(hparams)
        generated = model.gen_graph(hparams, vae)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(generated)

    def test_training_iseq(self):
        self._test_training("ISEQ")

    def test_generating_iseq(self):
        self._test_generating("ISEQ")

    def test_training_rnn(self):
        self._test_training("RNN")

    def test_generating_rnn(self):
        self._test_generating("RNN")

    def test_training_srnn(self):
        self._test_training("SRNN")

    def test_generating_srnn(self):
        self._test_generating("SRNN")


if __name__ == "__main__":
    tf.test.main()
