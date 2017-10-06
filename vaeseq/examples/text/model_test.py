"""Tests for training and generating graphs."""

import io
import os.path
import tensorflow as tf
from vaeseq import util

from vaeseq.examples.text import dataset as dataset_mod
from vaeseq.examples.text import hparams as hparams_mod
from vaeseq.examples.text import model


def _hparams(vae_type):
    """HParams used for testing the given VAE type."""
    hparams = hparams_mod.make_hparams()
    hparams.rnn_hidden_sizes = [4, 4]
    hparams.vae_type = vae_type
    hparams.check_numerics = True
    hparams.vocab_size = 5
    return hparams


def _observations(hparams):
    """Create observations for testing.."""
    temp_path = os.path.join(tf.test.get_temp_dir(), "corpus.txt")
    with io.open(temp_path, "w", encoding="utf-8") as temp_file:
        temp_file.write(u"1234567890" * 100)
    char_to_id, id_to_char = dataset_mod.vocabulary(temp_path,
                                                    hparams.vocab_size,
                                                    hparams.oov_buckets)
    dataset = dataset_mod.characters(temp_path,
                                     util.batch_size(hparams),
                                     util.sequence_size(hparams))
    return model.observations(dataset, char_to_id)


class ModelTest(tf.test.TestCase):

    def _test_training(self, vae_type):
        """Test the training graph for the given VAE type."""
        tf.set_random_seed(0)
        hparams = _hparams(vae_type)
        vae = model.make_vae(hparams)
        observed = _observations(hparams)
        train_op, debug_tensors = model.train_graph(hparams, vae, observed)
        debug_tensors["log_prob"] = model.eval_graph(hparams, vae, observed)
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(),
                      model.make_scaffold().local_init_op])
            for _ in range(100):
                # Warm up the log prob rolling average.
                sess.run(tf.get_collection("metric_updates"))
            debug_vals = sess.run(debug_tensors)
            self.assertGreaterEqual(debug_vals["divergence"], 0)
            for _ in range(100):
                sess.run([train_op] + tf.get_collection("metric_updates"))
            debug_vals2 = sess.run(debug_tensors)
            self.assertGreater(debug_vals2["elbo_opt"], debug_vals["elbo_opt"])
            self.assertGreater(debug_vals2["log_prob"], debug_vals["log_prob"])

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
