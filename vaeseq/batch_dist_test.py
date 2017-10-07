"""Tests for BatchDistribution."""

import numpy as np
import tensorflow as tf

from vaeseq import batch_dist


class BatchDistributionTest(tf.test.TestCase):

    def test_sample(self):
        dist = tf.distributions.Bernoulli(logits=tf.zeros((4, 5, 6)))
        batch_dist1 = batch_dist.BatchDistribution(dist)
        with self.test_session() as sess:
            sess.run(tf.assert_equal(dist.sample(seed=123),
                                     batch_dist1.sample(seed=123)))

    def test_log_prob(self):
        dist = tf.distributions.Bernoulli(logits=tf.zeros((4, 5, 6)))
        batch_dist1 = batch_dist.BatchDistribution(dist)
        batch_dist2 = batch_dist.BatchDistribution(dist, ndims=2)
        event = tf.zeros((4, 5, 6))
        with self.test_session() as sess:
            self.assertAllClose(sess.run(batch_dist1.log_prob(event)),
                                np.full((4, 5), 6 * np.log(0.5)))
            self.assertAllClose(sess.run(batch_dist2.log_prob(event)),
                                np.full((4,), 30 * np.log(0.5)))

    def test_prob(self):
        dist = tf.distributions.Bernoulli(probs=0.5 * tf.ones((4, 5, 6)))
        batch_dist1 = batch_dist.BatchDistribution(dist)
        batch_dist2 = batch_dist.BatchDistribution(dist, ndims=2)
        event = tf.zeros((4, 5, 6))
        with self.test_session() as sess:
            self.assertAllClose(sess.run(batch_dist1.prob(event)),
                                np.full((4, 5), 0.5 ** 6))
            self.assertAllClose(sess.run(batch_dist2.prob(event)),
                                np.full((4,), 0.5 ** 30))

    def test_is_scalar(self):
        dist = tf.distributions.Bernoulli(probs=0.5 * tf.ones((4, 5)))
        batch_dist1 = batch_dist.BatchDistribution(dist)
        batch_dist2 = batch_dist.BatchDistribution(dist, ndims=2)
        with self.test_session() as sess:
            self.assertAllEqual(
                sess.run([batch_dist1.is_scalar_event(),
                          batch_dist2.is_scalar_event()]),
                [False, False])
            self.assertAllEqual(
                sess.run([batch_dist1.is_scalar_batch(),
                          batch_dist2.is_scalar_batch()]),
                [False, True])


class GroupDistributionTest(tf.test.TestCase):

    def test_sample(self):
        components = {
            'dist_a': tf.distributions.Bernoulli(logits=tf.zeros(3)),
            'dist_b': tf.distributions.Normal(tf.zeros(3), tf.ones(3)),
        }
        dist = batch_dist.GroupDistribution(components)
        with self.test_session() as sess:
            val = sess.run(dist.sample())
            self.assertAllEqual(val['dist_a'].shape, [3])
            self.assertAllEqual(val['dist_b'].shape, [3])

    def test_log_prob(self):
        components = {
            'dist_a': tf.distributions.Bernoulli(logits=tf.zeros(3)),
            'dist_b': tf.distributions.Normal(tf.zeros(3), tf.ones(3)),
        }
        dist = batch_dist.GroupDistribution(components)
        with self.test_session() as sess:
            a_log_prob, b_log_prob, group_log_prob = sess.run([
                components['dist_a'].log_prob(0.),
                components['dist_b'].log_prob(0.),
                dist.log_prob({'dist_a': 0., 'dist_b': 0.})])
            self.assertAllClose(a_log_prob, np.log([0.5] * 3))
            self.assertAllClose(b_log_prob, np.log([(2 * np.pi) ** -0.5] * 3))
            self.assertAllClose(group_log_prob,
                                np.log([0.5 * (2 * np.pi) ** -0.5] * 3))

    def test_prob(self):
        components = {
            'dist_a': tf.distributions.Bernoulli(logits=tf.zeros(3)),
            'dist_b': tf.distributions.Normal(tf.zeros(3), tf.ones(3)),
        }
        dist = batch_dist.GroupDistribution(components)
        with self.test_session() as sess:
            a_prob, b_prob, group_prob = sess.run([
                components['dist_a'].prob(0.),
                components['dist_b'].prob(0.),
                dist.prob({'dist_a': 0., 'dist_b': 0.})])
            self.assertAllClose(a_prob, [0.5] * 3)
            self.assertAllClose(b_prob, [(2 * np.pi) ** -0.5] * 3)
            self.assertAllClose(group_prob, [0.5 * (2 * np.pi) ** -0.5] * 3)

    def test_is_scalar(self):
        assertions =  [
            dist.is_scalar_event() for dist in [
                batch_dist.GroupDistribution((((), ()))),
                batch_dist.GroupDistribution(
                    tf.distributions.Bernoulli(probs=0.5)),
                batch_dist.GroupDistribution(
                    tf.distributions.Bernoulli(probs=[0.5, 0.3])),
            ]
        ] + [
            tf.logical_not(dist.is_scalar_event()) for dist in [
                batch_dist.GroupDistribution(
                    (tf.distributions.Bernoulli(probs=0.5),
                     tf.distributions.Bernoulli(probs=0.5))),
                batch_dist.GroupDistribution(
                    tf.contrib.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(2), scale_diag=tf.ones(2))),
            ]
        ] + [
            dist.is_scalar_batch() for dist in [
                batch_dist.GroupDistribution((((), ()))),
                batch_dist.GroupDistribution(
                    tf.distributions.Bernoulli(probs=0.5)),
                batch_dist.GroupDistribution(
                    tf.contrib.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(2), scale_diag=tf.ones(2))),
            ]
        ] + [
            tf.logical_not(dist.is_scalar_batch()) for dist in [
                batch_dist.GroupDistribution(
                    (tf.distributions.Bernoulli(probs=0.5),
                     tf.distributions.Bernoulli(probs=0.5))),
                batch_dist.GroupDistribution(
                    tf.distributions.Bernoulli(probs=[0.5, 0.3])),
            ]
        ]
        with self.test_session() as sess:
            self.assertTrue(np.all(sess.run(assertions)))


if __name__ == '__main__':
    tf.test.main()
