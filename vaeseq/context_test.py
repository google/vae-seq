# Copyright 2018 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for context modules."""

import tensorflow as tf

from vaeseq import codec
from vaeseq import context as context_mod


class ContextTest(tf.test.TestCase):

    def testConstant(self):
        context = context_mod.as_context(tf.constant([[1,2,3]]))
        observed = tf.constant([["a", "b", "c", "d", "e"]])
        contexts = context.from_observations(observed)
        with self.test_session() as sess:
            contexts = sess.run(contexts)
            self.assertAllEqual(contexts, [[1, 2, 3, 0, 0]])

    def testEncodeObserved(self):
        encoder = codec.FlattenEncoder(input_size=tf.TensorShape([1]))
        context = context_mod.EncodeObserved(encoder)
        observed = tf.constant([[[1.], [2.], [3.]]])
        contexts = context.from_observations(observed)
        with self.test_session() as sess:
            contexts = sess.run(contexts)
            self.assertAllClose(contexts, [[[0.], [1.], [2.]]])

    def testChain(self):
        inputs = tf.constant([[[10.], [20.]]])
        observed = tf.constant([[[1.], [2.], [3.]]])
        encoder = codec.FlattenEncoder(input_size=tf.TensorShape([1]))
        context = context_mod.Chain([
            context_mod.as_context(inputs),
            context_mod.EncodeObserved(encoder, input_encoder=encoder),
        ])
        contexts = context.from_observations(observed)
        with self.test_session() as sess:
            inputs, contexts = sess.run(contexts)
            self.assertAllClose(inputs, [[[10.], [20.], [0.]]])
            self.assertAllClose(contexts, [[[0.], [1.], [2.]]])


if __name__ == '__main__':
    tf.test.main()
