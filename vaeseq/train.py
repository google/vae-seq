"""Training subgraph for a VAE."""

import sonnet as snt
import tensorflow as tf

from . import util


class Trainer(snt.AbstractModule):
    """This module produces a train_op given Agent contexts and observations."""

    def __init__(self, hparams, vae, name=None):
        super(Trainer, self).__init__(name=name)
        self._hparams = hparams
        self._vae = vae
        with self._enter_variable_scope():
            self._optimizer = tf.train.AdamOptimizer(hparams.learning_rate)

    def _build(self, contexts, observed):
        hparams = self._hparams
        latents, divs = self._vae.infer_latents(contexts, observed)
        log_probs = self._vae.log_prob_observed(contexts, latents, observed)

        # Compute the ELBO.
        batch_size = tf.to_float(tf.shape(log_probs)[0])
        log_prob = tf.reduce_sum(log_probs) / batch_size
        tf.summary.scalar("log_prob", log_prob)
        divergence = tf.reduce_sum(divs) / batch_size
        tf.summary.scalar("divergence", divergence)
        elbo = log_prob - divergence
        tf.summary.scalar("elbo", elbo)

        # We soften the divergence penalty at the start of training.
        divergence_strength = tf.sigmoid(
            tf.to_float(tf.train.get_or_create_global_step()) /
            hparams.divergence_strength_halfway_point - 1.)
        tf.summary.scalar("divergence_strength", divergence_strength)
        relaxed_elbo = log_prob - divergence * divergence_strength
        loss = -relaxed_elbo

        train_op = tf.contrib.training.create_train_op(
            loss,
            self._optimizer,
            summarize_gradients=True,
            check_numerics=hparams.check_numerics)

        debug_tensors = dict(
            log_prob=log_prob,
            divergence=divergence,
            elbo=elbo,
            loss=loss)
        return train_op, debug_tensors
