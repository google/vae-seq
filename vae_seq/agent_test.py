import sonnet as snt
import tensorflow as tf

from vae_seq import agent as agent_mod


class TestEnvironment(snt.RNNCore):

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


class IdentityObsEncoder(snt.AbstractModule):

    @property
    def output_size(self):
        return tf.TensorShape([1])

    def _build(self, x):
        return x


class AgentTest(tf.test.TestCase):

    def test_null_inputs(self):
        with self.test_session() as sess:
            self.assertAllEqual(
                sess.run(tf.shape(agent_mod.null_inputs(3, 5))), [3, 5, 0])

    def test_contexts_for_static_obs(self):
        agent = agent_mod.EncodeObsAgent(IdentityObsEncoder(name="identity"))
        obs = tf.constant([[[1.], [2.], [3.]],
                           [[4.], [5.], [6.]]])
        ctx = agent_mod.contexts_for_static_observations(
            obs, agent, agent_mod.null_inputs(2, 3))
        with self.test_session() as sess:
            self.assertAllClose(
                sess.run(ctx),
                [[[0.], [1.], [2.]],
                 [[0.], [4.], [5.]]])

    def test_contexts_from_env(self):
        env = TestEnvironment(name="test_env")
        agent = agent_mod.EncodeObsAgent(IdentityObsEncoder(name="identity"))
        ctx, obs = agent_mod.contexts_and_observations_from_environment(
            env, agent, agent_mod.null_inputs(2, 3))
        ctx2 = agent_mod.contexts_for_static_observations(
            obs, agent, agent_mod.null_inputs(2, 3))
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
