"""Training subgraph for a VAE."""

import abc
import functools
import sonnet as snt
import tensorflow as tf

from . import util


class TrainerBase(snt.AbstractModule):
    """Base class for Trainer objects, which construct a train op."""

    def __init__(self, name=None):
        super(TrainerBase, self).__init__(name=name)
        self._loss_applied = False

    @abc.abstractmethod
    def _apply_loss(self, contexts, observed):
        """Construct the objective and debug tensors."""

    def apply_loss(self, contexts, observed):
        loss, debug = self._apply_loss(contexts, observed)
        self._loss_applied = True
        return loss, debug

    @abc.abstractmethod
    def _train_op(self, loss):
        """Construct a train op from the loss tensor."""

    def _build(self, contexts, observed):
        loss, debug = self.apply_loss(contexts, observed)
        return self._train_op(loss), debug


class TrainerBaseWithOptimizer(TrainerBase):
    """Base class for Trainer objects that use an optimizer."""

    def __init__(self, hparams, vae, global_step, name=None):
        super(TrainerBaseWithOptimizer, self).__init__(name=name)
        self._hparams = hparams
        self._vae = vae
        self._global_step = global_step

    @abc.abstractmethod
    def _make_optimizer(self):
        """Construct the optimizer that produces the train op."""

    @abc.abstractmethod
    def _get_variables(self):
        """Returns the variables to optimize."""

    @util.lazy_property
    def optimizer(self):
        return self._make_optimizer()

    def _transform_gradients(self, gradients_to_variables):
        """Transform gradients before applying the optimizer."""
        if self._hparams.clip_gradient_norm > 0:
            gradients_to_variables = tf.contrib.training.clip_gradient_norms(
                gradients_to_variables,
                self._hparams.clip_gradient_norm)
        return gradients_to_variables

    def _train_op(self, loss):
        """Construct a train op from the loss tensor."""
        if not self._loss_applied:
            raise ValueError("You must apply the loss function first.")
        variables = self._get_variables()
        if variables is None:
            return tf.no_op()
        if not variables:
            raise ValueError("No trainable variables found.")
        return tf.contrib.training.create_train_op(
            loss,
            self.optimizer,
            global_step=self._global_step,
            variables_to_train=variables,
            transform_grads_fn=self._transform_gradients,
            summarize_gradients=True,
            check_numerics=self._hparams.check_numerics)


class AgentTrainer(TrainerBaseWithOptimizer):
    """Trainer for Agent variables."""

    def _make_optimizer(self):
        return tf.train.AdamOptimizer(self._hparams.agent_learning_rate)

    def _get_variables(self):
        agent_vars = self._vae.agent.get_variables()
        if agent_vars is None:
            return None
        agent_vars = set(snt.nest.flatten(agent_vars))
        trainable_vars = tf.trainable_variables()
        return [var for var in trainable_vars if var in agent_vars]

    def _apply_loss(self, contexts, observed):
        # We only check whether the agent knows how to extract
        # observation rewards here. Otherwise, we train on the
        # generated environment.
        del contexts  # Not used.
        agent = self._vae.agent
        observed_rewards = agent.rewards(observed)
        if observed_rewards is None:
            return 0., {}
        inputs = agent.get_inputs(util.batch_size(self._hparams),
                                  util.sequence_size(self._hparams))
        generated, log_probs = self._vae.gen_log_probs_core.generate(inputs)[0]
        return AgentLoss(self._hparams, agent)(generated, log_probs)


class VAETrainer(TrainerBaseWithOptimizer):
    """Trainer for non-Agent variables."""

    def _make_optimizer(self):
        return tf.train.AdamOptimizer(self._hparams.learning_rate)

    def _get_variables(self):
        trainable_vars = tf.trainable_variables()
        agent_vars = self._vae.agent.get_variables()
        if agent_vars is None:
            return trainable_vars
        agent_vars = set(snt.nest.flatten(agent_vars))
        return [var for var in trainable_vars if var not in agent_vars]

    def _apply_loss(self, contexts, observed):
        return ELBOLoss(self._hparams, self._vae)(contexts, observed)


class Trainer(TrainerBase):
    """Train both the VAE and the Agent."""

    def __init__(self, hparams, vae, global_step, name=None):
        super(TrainerBase, self).__init__(name=name)
        self._hparams = hparams
        self._vae = vae
        self._global_step = global_step
        with self._enter_variable_scope():
            self._vae_trainer = VAETrainer(hparams, vae, global_step)
            self._agent_trainer = AgentTrainer(hparams, vae, global_step=None)

    def _apply_loss(self, contexts, observed):
        agent_loss, debug = self._agent_trainer.apply_loss(contexts, observed)
        vae_loss, vae_debug = self._vae_trainer.apply_loss(contexts, observed)
        debug.update(vae_debug)
        return (agent_loss, vae_loss), debug

    def _train_op(self, loss):
        agent_loss, vae_loss = loss
        return tf.group(self._agent_trainer._train_op(agent_loss),
                        self._vae_trainer._train_op(vae_loss))


class AgentLoss(snt.AbstractModule):
    """Calculates an objective for training the Agent to maximize rewards."""

    def __init__(self, hparams, agent, name=None):
        super(AgentLoss, self).__init__(name=name)
        self._hparams = hparams
        self._agent = agent

    def _build(self, observed, log_probs):
        debug_tensors = {}
        scalar_summary = functools.partial(_scalar_summary, debug_tensors)
        rewards = self._agent.rewards(observed)
        assert rewards is not None

        # Apply batch normalization to the rewards to simplify training.
        batch_norm = snt.BatchNorm(axis=[0, 1])
        rewards = batch_norm(rewards, is_training=True)
        scalar_summary("mean_reward", tf.squeeze(batch_norm.moving_mean))
        loss = -_sum_time_average_batch(rewards)

        if self._hparams.reinforce_agent_across_timesteps:
            # Since observations are fed back into the following
            # timesteps, propagate gradient across observations using the
            # log-derivative trick (REINFORCE).
            cumulative_rewards = tf.cumsum(rewards, axis=1, reverse=True)
            proxy_rewards = log_probs * tf.stop_gradient(cumulative_rewards)
            mean_proxy_reward = _sum_time_average_batch(proxy_rewards)
            scalar_summary("mean_proxy_reward", mean_proxy_reward)
            loss -= mean_proxy_reward

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
