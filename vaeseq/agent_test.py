"""Tests for Agent functionality."""

import tensorflow as tf

from vaeseq import agent as agent_mod
from vaeseq import codec


class TestEnvironment(agent_mod.Environment):
    """An environment that returns observations as context + state."""

    @property
    def output_size(self):
        return tf.TensorShape([1])

    @property
    def output_dtype(self):
        return tf.float32

    @property
    def state_size(self):
        return tf.TensorShape([])

    def initial_state(self, batch_size):
        return tf.to_float(tf.range(1, batch_size + 1))

    def _build(self, context, state):
        return context + tf.expand_dims(state, 1), state


def _make_agent():
    """Returns an agent that passes through the last observation as context."""
    return agent_mod.EncodeObsAgent(codec.FlattenObsEncoder(input_size=1))


class AgentTest(tf.test.TestCase):

    def test_null_inputs(self):
        with self.test_session() as sess:
            null_inputs = _make_agent().get_inputs(3, 5)
            self.assertAllEqual(
                sess.run(tf.shape(null_inputs)), [3, 5, 0])

    def test_contexts_for_static_obs(self):
        obs = tf.constant([[[1.], [2.], [3.]],
                           [[4.], [5.], [6.]]])
        agent = _make_agent()
        ctx = agent.contexts_for_static_observations(obs)
        with self.test_session() as sess:
            self.assertAllClose(
                sess.run(ctx),
                [[[0.], [1.], [2.]],
                 [[0.], [4.], [5.]]])

    def test_contexts_from_env(self):
        env = TestEnvironment(name="test_env")
        agent = _make_agent()
        agent_input = agent.get_inputs(2, 3)
        ctx, _unused_actions, obs = agent.run_environment(env, agent_input)
        ctx2 = agent.contexts_for_static_observations(obs, agent_input)
        with self.test_session() as sess:
            vals = sess.run(dict(ctx=ctx, ctx2=ctx2, obs=obs))
        self.assertAllClose(vals['ctx'], vals['ctx2'])
        self.assertAllClose(vals['ctx'],
                            [[[0.], [1.], [2.]],
                             [[0.], [2.], [4.]]])
        self.assertAllClose(vals['obs'],
                            [[[1.], [2.], [3.]],
                             [[2.], [4.], [6.]]])

if __name__ == '__main__':
    tf.test.main()
