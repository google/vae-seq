"""Tests for BatchDistribution."""

import numpy as np
import tensorflow as tf

from vaeseq import batch_distribution


class BatchDistributionTest(tf.test.TestCase):

    def test_sample(self):
        dist = tf.distributions.Bernoulli(logits=tf.zeros((4, 5, 6)))
        batch_dist = batch_distribution.BatchDistribution(dist)
        with self.test_session() as sess:
            sess.run(tf.assert_equal(dist.sample(seed=123),
                                     batch_dist.sample(seed=123)))

    def test_log_prob(self):
        dist = tf.distributions.Bernoulli(logits=tf.zeros((4, 5, 6)))
        batch_dist1 = batch_distribution.BatchDistribution(dist)
        batch_dist2 = batch_distribution.BatchDistribution(dist, ndims=2)
        event = tf.zeros((4, 5, 6))
        with self.test_session() as sess:
            self.assertAllClose(sess.run(batch_dist1.log_prob(event)),
                                np.full((4, 5), 6 * np.log(0.5)))
            self.assertAllClose(sess.run(batch_dist2.log_prob(event)),
                                np.full((4,), 30 * np.log(0.5)))

    def test_prob(self):
        dist = tf.distributions.Bernoulli(probs=0.5 * tf.ones((4, 5, 6)))
        batch_dist1 = batch_distribution.BatchDistribution(dist)
        batch_dist2 = batch_distribution.BatchDistribution(dist, ndims=2)
        event = tf.zeros((4, 5, 6))
        with self.test_session() as sess:
            self.assertAllClose(sess.run(batch_dist1.prob(event)),
                                np.full((4, 5), 0.5 ** 6))
            self.assertAllClose(sess.run(batch_dist2.prob(event)),
                                np.full((4,), 0.5 ** 30))


if __name__ == '__main__':
    tf.test.main()
