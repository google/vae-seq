"""Training subgraph for a VAE."""

import functools
import sonnet as snt
import tensorflow as tf

from . import util


class Trainer(snt.AbstractModule):
    """Returns a train op that minimizes the given loss."""

    def __init__(self, hparams, vae, name=None):
        super(Trainer, self).__init__(name=name)
        self._hparams = hparams
        self._vae = vae
        with self._enter_variable_scope():
            self._optimizer = tf.train.AdamOptimizer(hparams.learning_rate)

    def _transform_gradients(self, gradients_to_variables):
        if self._hparams.clip_gradient_norm > 0:
            gradients_to_variables = tf.contrib.training.clip_gradient_norms(
                gradients_to_variables,
                self._hparams.clip_gradient_norm)
        return gradients_to_variables

    def _trainable_variables(self, in_agent):
        """Train only-or everything but-the Agent."""
        trainable_vars = tf.trainable_variables()
        agent_vars = set(snt.nest.flatten(self._vae.agent.agent_variables()))
        if in_agent:
            return [var for var in trainable_vars if var in agent_vars]
        return [var for var in trainable_vars if var not in agent_vars]

    def _create_train_op(self, loss, in_agent):
        """Creates a train op."""
        return tf.contrib.training.create_train_op(
            loss,
            self._optimizer,
            variables_to_train=self._trainable_variables(in_agent=in_agent),
            transform_grads_fn=self._transform_gradients,
            summarize_gradients=True,
            check_numerics=self._hparams.check_numerics)

    def _build(self, elbo_loss, agent_loss=None):
        train_op = self._create_train_op(elbo_loss, in_agent=False)
        if agent_loss is not None:
            train_op = tf.group(
                train_op,
                self._create_train_op(agent_loss, in_agent=True))
        return train_op


class AgentLoss(snt.AbstractModule):
    """Calculates an objective for training the Agent to maximize rewards."""

    def __init__(self, hparams, vae, name=None):
        super(AgentLoss, self).__init__(name=name)
        self._hparams = hparams
        self._vae = vae

    def _build(self, observed, log_probs):
        debug_tensors = {}
        scalar_summary = functools.partial(_scalar_summary, debug_tensors)

        rewards = self._vae.agent.rewards(observed)
        assert rewards is not None

        # Apply batch normalization to the rewards to simplify training.
        batch_norm = snt.BatchNorm(axis=[0, 1])
        rewards = batch_norm(rewards, is_training=True)
        scalar_summary("mean_reward", tf.squeeze(batch_norm.moving_mean))

        # Since observations are fed back into the following
        # timesteps, propagate gradient across observations using the
        # log-derivative trick (REINFORCE).
        cumulative_rewards = tf.cumsum(rewards, axis=1, reverse=True)
        proxy_rewards = log_probs * tf.stop_gradient(cumulative_rewards)

        mean_proxy_reward = _sum_time_average_batch(proxy_rewards)
        scalar_summary("mean_proxy_reward", mean_proxy_reward)

        loss = -(mean_proxy_reward + _sum_time_average_batch(rewards))
        scalar_summary("agent_loss", loss)
        return loss, debug_tensors


class ELBOLoss(snt.AbstractModule):
    """Calculates an objective for maximizing the evidence lower bound."""

    def __init__(self, hparams, vae, name=None):
        super(ELBOLoss, self).__init__(name=name)
        self._hparams = hparams
        self._vae = vae

    def _build(self, contexts, observed):
        debug_tensors = {}
        scalar_summary = functools.partial(_scalar_summary, debug_tensors)

        latents, divs = self._vae.infer_latents(contexts, observed)
        log_probs = self._vae.log_prob_observed(contexts, latents, observed)
        log_prob = _sum_time_average_batch(log_probs)
        divergence = _sum_time_average_batch(divs)
        scalar_summary("log_prob", log_prob)
        scalar_summary("divergence", divergence)
        scalar_summary("ELBO", log_prob - divergence)

        # We soften the divergence penalty at the start of training.
        divergence_strength = tf.sigmoid(
            tf.to_float(tf.train.get_or_create_global_step()) /
            self._hparams.divergence_strength_halfway_point - 1.)
        scalar_summary("divergence_strength", divergence_strength)
        relaxed_elbo = log_prob - divergence * divergence_strength
        loss = -relaxed_elbo
        scalar_summary("elbo_loss", loss)
        return loss, debug_tensors


def _scalar_summary(debug_tensors, name, tensor):
    """Add a summary and a debug output tensor."""
    tensor = tf.convert_to_tensor(tensor, name=name)
    debug_tensors[name] = tensor
    tf.summary.scalar(name, tensor)


def _sum_time_average_batch(tensor):
    """Sum across time and average over batch entries."""
    return tf.reduce_mean(tf.reduce_sum(tensor, axis=1), axis=0)
