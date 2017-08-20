import sonnet as snt
import tensorflow as tf


class TrainOps(snt.AbstractModule):
    """Training subgraph for a VAE."""

    def __init__(self, hparams, vae, name=None):
        super(TrainOps, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._vae = vae

    def _build(self, contexts, observed):
        hparams = self._hparams
        latents, divs = self._vae.infer_latents(contexts, observed)
        log_probs = self._vae.log_prob_observed(contexts, latents, observed)

        # Compute the ELBO.
        log_prob = tf.reduce_sum(log_probs) / hparams.batch_size
        tf.summary.scalar("log_prob", log_prob)
        divergence = tf.reduce_sum(divs) / hparams.batch_size
        tf.summary.scalar("divergence", divergence)
        elbo = log_prob - divergence
        tf.summary.scalar("elbo", elbo)

        # We soften the divergence penalty at the start of training.
        divergence_strength = tf.sigmoid(
            tf.to_float(tf.train.get_or_create_global_step()) /
            hparams.divergence_strength_halfway_point - 1.)
        tf.summary.scalar("divergence_strength", divergence_strength)
        elbo_opt = log_prob - divergence * divergence_strength

        # Compute gradients.
        optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
        grads_and_vars = optimizer.compute_gradients(-elbo_opt)
        for grad, var in grads_and_vars:
            tag = var.name.replace(":0", "")
            if grad is None:
                print "WARNING: Gradient for " + tag + " is missing!"
                continue
            tf.summary.histogram(tag, var)
            tf.summary.histogram(tag + "/gradient", grad)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=tf.train.get_or_create_global_step())
        if hparams.check_numerics and False:  # FIXME
            deps = [tf.add_check_numerics_ops(), train_op]
            with tf.control_dependencies(deps):
                train_op = tf.no_op()

        debug_tensors = dict(
            log_prob=log_prob,
            divergence=divergence,
            elbo=elbo,
            elbo_opt=elbo_opt)
        return train_op, debug_tensors
