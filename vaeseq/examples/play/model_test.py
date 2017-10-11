"""Tests for training and generating graphs."""

import tensorflow as tf
from vaeseq import model_test

from vaeseq.examples.play import hparams as hparams_mod
from vaeseq.examples.play import model as model_mod


class ModelTest(model_test.ModelTest):

    def _setup_model(self, session_params):
        self.train_dataset = True
        self.valid_dataset = None
        self.hparams = hparams_mod.make_hparams(rnn_hidden_sizes=[4, 4],
                                                check_numerics=True)
        self.model = model_mod.Model(self.hparams, session_params)


if __name__ == "__main__":
    tf.test.main()
