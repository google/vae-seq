"""Tests for ModelBase."""

import itertools
import os.path
import tensorflow as tf

from vaeseq import agent as agent_mod
from vaeseq import codec
from vaeseq import hparams as hparams_mod
from vaeseq import model as model_mod
from vaeseq import util
from vaeseq import vae as vae_mod


class ModelTest(tf.test.TestCase):

    def setUp(self):
        super(ModelTest, self).setUp()
        log_dir = os.path.join(self.get_temp_dir(), "log_dir")
        session_params = model_mod.ModelBase.SessionParams(log_dir=log_dir)
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

    def test_training_and_eval(self):
        train_debug1 = self._train()
        eval_debug1 = self._evaluate()
        train_debug2 = self._train()
        eval_debug2 = self._evaluate()
        self.assertLess(train_debug2["loss"], train_debug1["loss"])
        self.assertGreater(eval_debug2["log_prob"], eval_debug1["log_prob"])

    def test_genaration(self):
        # Just make sure the graph executes without error.
        for seq in itertools.islice(self.model.generate(), 10):
            tf.logging.debug("Generated: %r", seq)


class MockModel(model_mod.ModelBase):
    """Modeling zeros for testing."""

    def _make_vae(self):
        obs_encoder = codec.MLPObsEncoder(self.hparams)
        obs_decoder = codec.BatchDecoder(
            codec.MLPObsDecoder(
                self.hparams,
                codec.NormalDecoder(self.hparams),
                param_size=4),
            event_size=[2])
        agent = agent_mod.EncodeObsAgent(obs_encoder)
        return vae_mod.make(self.hparams, agent, obs_encoder, obs_decoder)

    def _agent_inputs(self, batch_size, sequence_size):
        return agent_mod.null_inputs(batch_size, sequence_size)

    def _open_dataset(self, dataset):
        batch_size = util.batch_size(self.hparams)
        sequence_size = util.sequence_size(self.hparams)
        observed = tf.zeros([batch_size, sequence_size, 2])
        contexts = agent_mod.contexts_for_static_observations(
            observed,
            self.vae.agent,
            agent_inputs=self._agent_inputs(batch_size, sequence_size))
        return contexts, observed

    def _make_output_summary(self, tag, observed):
        return tf.summary.histogram(tag, observed, collections=[])


if __name__ == "__main__":
    tf.test.main()
