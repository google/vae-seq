"""Tests for context modules."""

import tensorflow as tf

from vaeseq import codec
from vaeseq import context as context_mod


class ContextTest(tf.test.TestCase):

    def testConstant(self):
        observed = tf.constant([["a", "b", "c", "d", "e"]])
        context = context_mod.Constant(tf.constant([[1,2,3]]))
        contexts = context.from_observations(inputs=None, observed=observed)
        with self.test_session() as sess:
            contexts = sess.run(contexts)
            self.assertAllEqual(contexts, [[1, 2, 3, 0, 0]])

    def testEncodeObserved(self):
        observed = tf.constant([[[1.], [2.], [3.]]])
        encoder = codec.FlattenObsEncoder(input_size=tf.TensorShape([1]))
        context = context_mod.EncodeObserved(encoder)
        contexts = context.from_observations(inputs=None, observed=observed)
        with self.test_session() as sess:
            contexts = sess.run(contexts)
            self.assertAllClose(contexts, [[[0.], [1.], [2.]]])

    def testChain(self):
        inputs = tf.constant([[[10.], [20.]]])
        observed = tf.constant([[[1.], [2.], [3.]]])
        encoder = codec.FlattenObsEncoder(input_size=tf.TensorShape([1]))
        context = context_mod.Chain([
            context_mod.Constant(inputs),
            context_mod.EncodeObserved(encoder, input_encoder=encoder),
        ])
        contexts = context.from_observations(inputs=None, observed=observed)
        with self.test_session() as sess:
            inputs, contexts = sess.run(contexts)
            self.assertAllClose(inputs, [[[10.], [20.], [0.]]])
            self.assertAllClose(contexts, [[[0.], [1.], [2.]]])


if __name__ == '__main__':
    tf.test.main()
