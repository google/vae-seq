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
            self._ema = ExponentialMovingAverage()
            self._optimizer = tf.train.AdamOptimizer(hparams.learning_rate)

    def _transform_gradients(self, gradients_to_variables):
        if self._hparams.clip_gradient_norm > 0:
            gradients_to_variables = tf.contrib.training.clip_gradient_norms(
                gradients_to_variables,
                self._hparams.clip_gradient_norm)
        return gradients_to_variables

    def _trainable_variables(self, in_agent):
        trainable_vars = tf.trainable_variables()
        agent_vars = set(snt.nest.flatten(self._vae.agent.agent_variables()))
        if in_agent:
            return [var for var in trainable_vars if var in agent_vars]
        return [var for var in trainable_vars if var not in agent_vars]

    def _build(self, contexts, observed, rewards=None):
        hparams = self._hparams
        latents, divs = self._vae.infer_latents(contexts, observed)

        debug_tensors = {}
        def _scalar_summary(name, tensor):
            """Add a summary and a debug output tensor."""
            tensor = tf.convert_to_tensor(tensor, name=name)
            debug_tensors[name] = tensor
            tf.summary.scalar(name, tensor)

        def _sum_time_average_batch(tensor):
            return tf.reduce_mean(tf.reduce_sum(tensor, axis=1), axis=0)

        # Compute the ELBO.
        log_probs = self._vae.log_prob_observed(contexts, latents, observed)
        log_prob = _sum_time_average_batch(log_probs)
        divergence = _sum_time_average_batch(divs)
        _scalar_summary("log_prob", log_prob)
        _scalar_summary("divergence", divergence)
        _scalar_summary("ELBO", log_prob - divergence)
        # We soften the divergence penalty at the start of training.
        divergence_strength = tf.sigmoid(
            tf.to_float(tf.train.get_or_create_global_step()) /
            hparams.divergence_strength_halfway_point - 1.)
        _scalar_summary("divergence_strength", divergence_strength)
        relaxed_elbo = log_prob - divergence * divergence_strength
        elbo_loss = -relaxed_elbo
        _scalar_summary("elbo_loss", elbo_loss)

        loss = elbo_loss
        train_op = tf.contrib.training.create_train_op(
            elbo_loss,
            self._optimizer,
            variables_to_train=self._trainable_variables(in_agent=False),
            transform_grads_fn=self._transform_gradients,
            summarize_gradients=True,
            check_numerics=hparams.check_numerics)

        # Compute the reward signal via REINFORCE.
        if rewards is not None:
            cumulative_rewards = tf.reduce_mean(
                tf.cumsum(rewards, axis=1, reverse=True),
                axis=0)
            _scalar_summary("total_reward", cumulative_rewards[0])
            if hparams.use_control_variates:
                # Subtract a mean of rewards to decrease variance.
                control_variate = tf.reduce_mean(cumulative_rewards)
                if hparams.control_variates_ema_decay > 0:
                    # Use a rolling average of the cumulative rewards.
                    control_variate = self._ema(control_variate)
                _scalar_summary("reward_control_variate", control_variate)
                cumulative_rewards -= control_variate
            # Recompute the log-probs with gradient only going to the
            # agent via the contexts.
            log_probs_stopgrad = self._vae.log_prob_observed(
                contexts,
                snt.nest.map(tf.stop_gradient, latents),
                snt.nest.map(tf.stop_gradient, observed))
            # Don't try to increase the reward directly.
            cumulative_rewards_stopgrad = tf.stop_gradient(cumulative_rewards)
            reinforce_loss = _sum_time_average_batch(
                -cumulative_rewards_stopgrad * log_probs_stopgrad)
            _scalar_summary("reinforce_loss", reinforce_loss)
            loss += reinforce_loss
            reinforce_train_op = tf.contrib.training.create_train_op(
                reinforce_loss,
                self._optimizer,
                variables_to_train=self._trainable_variables(in_agent=True),
                transform_grads_fn=self._transform_gradients,
                summarize_gradients=True,
                check_numerics=hparams.check_numerics)
            train_op = tf.group(train_op, reinforce_train_op)

        _scalar_summary("loss", loss)
        return train_op, debug_tensors


class ExponentialMovingAverage(snt.AbstractModule):
    """Simple replacement for tf.train.ExponentialMovingAverage for Sonnet."""

    def __init__(self, name=None):
        super(ExponentialMovingAverage, self).__init__(name=name)

    def _build(self, value, decay=0.99):
        avg = tf.get_local_variable(
            name=self.module_name + "_avg",
            shape=value.get_shape(),
            dtype=value.dtype,
            initializer=tf.zeros_initializer())
        delta = (avg - value) * decay
        return tf.assign_sub(avg, delta)
