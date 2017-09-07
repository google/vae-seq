# Copyright 2017 Google, Inc.,
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

"""Tests for utility functions in util.py."""

import numpy as np
import tensorflow as tf

from vae_seq import hparams as hparams_mod
from vae_seq import util


def _add_sub_core():
    """Creates an RNN Core: (a, b, c), s -> (a+s, b-s, c+s), -s"""
    return util.WrapRNNCore(
        lambda inp, state: (inp + state, -state),
        state_size=tf.TensorShape([]),
        output_size=tf.TensorShape([1]),
        name="AddSubCore")


def _identity_core(input_shape):
    """Creates an RNN Core that just propagates its inputs."""
    return util.WrapRNNCore(
        lambda inp, _: (inp, ()),
        state_size=(),
        output_size=input_shape,
        name="IdentityCore")


class UtilTest(tf.test.TestCase):

    def test_calc_kl_analytical(self):
        hparams = hparams_mod.make_hparams(use_monte_carlo_kl=False)
        dist_a = tf.distributions.Bernoulli(probs=0.5)
        dist_b = tf.distributions.Bernoulli(probs=0.3)
        kl_div = util.calc_kl(hparams, dist_a.sample(), dist_a, dist_b)
        with self.test_session():
            self.assertAllClose(
                kl_div.eval(),
                0.5 * (np.log(0.5 / 0.3) + np.log(0.5 / 0.7)))

    def test_calc_kl_mc(self):
        tf.set_random_seed(0)
        hparams = hparams_mod.make_hparams(use_monte_carlo_kl=True)
        samples = 1000
        dist_a = tf.distributions.Bernoulli(probs=tf.fill([samples], 0.5))
        dist_b = tf.distributions.Bernoulli(probs=tf.fill([samples], 0.3))
        kl_div = tf.reduce_mean(
            util.calc_kl(hparams, dist_a.sample(), dist_a, dist_b),
            axis=0)
        with self.test_session():
            self.assertAllClose(
                kl_div.eval(),
                0.5 * (np.log(0.5 / 0.3) + np.log(0.5 / 0.7)),
                atol=0.05)

    def test_concat_features(self):
        feature1 = tf.constant([[1, 2]])
        feature2 = tf.constant([[3]])
        feature3 = tf.constant([[4, 5]])
        with self.test_session():
            self.assertAllEqual(
                util.concat_features((feature1, (feature2, feature3))).eval(),
                [[1, 2, 3, 4, 5]])

    def test_wrap_rnn_core(self):
        core = _add_sub_core()
        input_ = tf.constant([[[1], [2], [3]]])
        state = tf.constant(5)
        output, out_state = tf.nn.dynamic_rnn(core, input_, initial_state=state)
        with self.test_session():
            self.assertEqual(out_state.eval(), -5)
            self.assertAllEqual(output.eval(), [[[1 + 5], [2 - 5], [3 + 5]]])

    def test_reverse_dynamic_rnn(self):
        core = _add_sub_core()
        input_ = tf.constant([[[1], [2]]])
        state = tf.constant(5)
        output, _ = util.reverse_dynamic_rnn(
            core, input_, initial_state=state)
        with self.test_session():
            self.assertAllEqual(output.eval(), [[[1 - 5], [2 + 5]]])

    def test_heterogeneous_dynamic_rnn(self):
        inputs = (tf.constant([[["hi"], ["there"]]]),
                  tf.constant([[[1, 2], [3, 4]]], dtype=tf.int32))
        core = _identity_core((tf.TensorShape([1]), tf.TensorShape([2])))
        outputs, _ = util.heterogeneous_dynamic_rnn(
            core, inputs, initial_state=(), output_dtypes=(tf.string, tf.int32))
        with self.test_session() as sess:
            inputs, outputs = sess.run((inputs, outputs))
            self.assertAllEqual(inputs[0], outputs[0])
            self.assertAllEqual(inputs[1], outputs[1])


if __name__ == "__main__":
    tf.test.main()
