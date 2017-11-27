"""Tests for ModelBase."""

import itertools
import os.path
import tensorflow as tf

from vaeseq import codec
from vaeseq import hparams as hparams_mod
from vaeseq import model as model_mod
from vaeseq import util


class ModelTest(tf.test.TestCase):

    def setUp(self):
        super(ModelTest, self).setUp()
        log_dir = os.path.join(self.get_temp_dir(), "log_dir")
        session_config = tf.ConfigProto()
        session_config.device_count["GPU"] = 0
        session_params = model_mod.ModelBase.SessionParams(
            log_dir=log_dir, session_config=session_config)
        self._setup_model(session_params)

    def _setup_model(self, session_params):
        self.train_dataset = "train"
        self.valid_dataset = "valid"
        self.hparams = hparams_mod.make_hparams()
        self.model = MockModel(self.hparams, session_params)

    def _train(self):
        return self.model.train(self.train_dataset, num_steps=20,
                                valid_dataset=self.valid_dataset)

    def _evaluate(self):
        return self.model.evaluate(self.train_dataset, num_steps=20)

    def test_inputs(self):
        train_inputs, observed = self.model.dataset(
            self.train_dataset)
        self.assertEqual(train_inputs is None, self.model.inputs is None)
        if self.model.inputs is not None:
            gen_inputs = self.model.inputs.from_observations(inputs=None,
                                                             observed=observed)
            self.assertEqual(gen_inputs.dtype, train_inputs.dtype)
            gen_inputs.get_shape().assert_is_compatible_with(
                train_inputs.get_shape())

    def test_training_and_eval(self):
        train_debug1 = self._train()
        eval_debug1 = self._evaluate()
        train_debug2 = self._train()
        eval_debug2 = self._evaluate()
        self.assertLess(train_debug2["elbo_loss"], train_debug1["elbo_loss"])
        self.assertGreater(eval_debug2["log_prob"], eval_debug1["log_prob"])

    def test_genaration(self):
        # Just make sure the graph executes without error.
        for seq in itertools.islice(self.model.generate(), 10):
            tf.logging.debug("Generated: %r", seq)


class MockModel(model_mod.ModelBase):
    """Modeling zeros for testing."""

    def _make_encoder(self):
        return codec.MLPObsEncoder(self.hparams)

    def _make_decoder(self):
        return codec.BatchDecoder(
            codec.MLPObsDecoder(
                self.hparams,
                codec.NormalDecoder(self.hparams),
                param_size=4),
            event_size=[2])

    def _make_dataset(self, dataset):
        batch_size = util.batch_size(self.hparams)
        sequence_size = util.sequence_size(self.hparams)
        observed = tf.zeros([batch_size, sequence_size, 2])
        inputs = None
        return inputs, observed

    def _make_output_summary(self, tag, observed):
        return tf.summary.histogram(tag, observed, collections=[])


if __name__ == "__main__":
    tf.test.main()
