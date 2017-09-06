"""Tests for Agent functionality."""

import numpy as np
import tensorflow as tf
from vae_seq import util

from examples.music_net import agent as agent_mod
from examples.music_net import codec
from examples.music_net import hparams as hparams_mod


def _make_hparams():
    """Returns HParams used for testing."""
    return hparams_mod.make_hparams(
        batch_size=2,
        sequence_size=3,
        samples_per_step=200,
        audio_rate=13)

def _make_agent(hparams):
    """Returns an instance of agent_mod.Agent."""
    obs_encoder = codec.AudioObsEncoder(hparams)
    return agent_mod.Agent(hparams, obs_encoder)


class AgentTest(tf.test.TestCase):

    def test_timing_input(self):
        hparams = _make_hparams()
        inputs = agent_mod.timing_input(hparams, offsets=[[0], [100]])
        with self.test_session() as sess:
            self.assertAllEqual(
                sess.run(inputs),
                [[[0], [200], [400]],
                 [[100], [300], [500]]])

    def test_agent_context(self):
        hparams = _make_hparams()
        agent = _make_agent(hparams)
        state = agent.initial_state(util.batch_size(hparams))
        samples = tf.to_float(tf.stack([tf.range(0, 200), tf.range(400, 600)]))
        offsets = [[200], [300]]
        state = agent.observe(offsets, samples, state)
        context = agent.context(offsets, state)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            metronome, prev_obs_enc = sess.run(context)
            self.assertAllEqual(prev_obs_enc.shape,
                                [hparams.batch_size,
                                 hparams.obs_encoder_fc_layers[-1]])
            self.assertAllClose(
                metronome,
                np.sin(np.array(offsets, dtype=np.float64) * 2 * np.pi
                       / hparams.audio_rate),
                atol=1e-4)


if __name__ == "__main__":
    tf.test.main()
