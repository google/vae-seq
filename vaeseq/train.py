"""Training subgraph for a VAE."""

import abc
import functools
import numpy as np
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
        agent = self._vae.agent
        observed_rewards = agent.rewards(observed)
        if observed_rewards is None:
            return 0., {}

        loss_fn = AgentLoss(self._hparams, agent)
        if self._hparams.train_agent_from_model:
            del contexts  # Not used.
            generated, log_probs = self._vae.gen_log_probs_core.generate(
                agent.get_inputs(util.batch_size(self._hparams),
                                 util.sequence_size(self._hparams)))[0]
            return loss_fn(generated, log_probs)
        latents, _unused_divs = self._vae.infer_latents(contexts, observed)
        log_probs = self._vae.log_prob_observed(contexts, latents, observed)
        return loss_fn(observed, log_probs)


class VAETrainer(TrainerBaseWithOptimizer):
    """Trainer for non-Agent variables."""

    @util.lazy_property
    def replay_buffer(self):
        with self._enter_variable_scope():
            return ReplayBuffer(
                capacity=self._hparams.replay_buffer,
                types=(self._vae.agent.context_dtype,
                       self._vae.observed_distcore.event_dtype),
                sizes=(self._vae.agent.context_size,
                       self._vae.observed_distcore.event_size))

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
        loss_fn = ELBOLoss(self._hparams, self._vae)
        loss, debug = loss_fn(contexts, observed)
        if self._hparams.replay_buffer > 0:
            replay_ctx, replay_obs = self.replay_buffer((contexts, observed))
            replay_loss, _unused_debug = loss_fn(replay_ctx, replay_obs)
            loss = 0.5 * (loss + replay_loss)
        return loss, debug


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

        mean_reward = tf.reduce_mean(rewards)
        scalar_summary("mean_reward", mean_reward)
        loss = -mean_reward

        if self._hparams.reinforce_agent_across_timesteps:
            # Since observations are fed back into the following
            # timesteps, propagate gradient across observations using the
            # log-derivative trick (REINFORCE).
            cumulative_rewards = tf.cumsum(rewards, axis=-1, reverse=True)
            proxy_rewards = log_probs * tf.stop_gradient(cumulative_rewards)
            mean_proxy_reward = tf.reduce_mean(proxy_rewards)
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
        log_prob = tf.reduce_mean(log_probs)
        divergence = tf.reduce_mean(divs)
        scalar_summary("log_prob", log_prob)
        scalar_summary("divergence", divergence)
        scalar_summary("ELBO", log_prob - divergence)

        # We soften the divergence penalty at the start of training.
        temp_start = -np.log(self._hparams.divergence_strength_start)
        temp_decay = ((-np.log(0.5) / temp_start) **
                      (1. / self._hparams.divergence_strength_half))
        global_step = tf.to_double(tf.train.get_or_create_global_step())
        divergence_strength = tf.to_float(
            tf.exp(-temp_start * tf.pow(temp_decay, global_step)))
        scalar_summary("divergence_strength", divergence_strength)
        relaxed_elbo = log_prob - divergence * divergence_strength

        loss = -relaxed_elbo
        scalar_summary("elbo_loss", loss)
        return loss, debug_tensors


class ReplayBuffer(snt.AbstractModule):
    """A simple replay buffer for sequential data."""

    def __init__(self, capacity, types, sizes, name=None):
        super(ReplayBuffer, self).__init__(name=name)
        self._capacity = capacity
        self._types = types
        self._sizes = sizes
        self._queue = tf.PriorityQueue(
            capacity=1 << 30,  # Queueing over capacity causes blocking.
            types=[tf.string] * len(snt.nest.flatten(self._types)),
            shapes=[tf.TensorShape([3])] * len(snt.nest.flatten(self._sizes)),
            name=self.module_name + "/queue")

    def _build(self, next_sample):
        flat_sample = snt.nest.flatten(next_sample)
        batch_size = tf.shape(flat_sample[0])[0]
        priorities = tf.random_uniform(
            shape=[batch_size], maxval=1 << 62, dtype=tf.int64,
            name="replay_priorities")
        enqueue_data = [priorities] + [
            tf.serialize_many_sparse(_dense_to_sparse(component))
            for component in flat_sample]
        enqueue_op = self._queue.enqueue_many(enqueue_data)
        with tf.control_dependencies([enqueue_op]):
            over_cap = self._queue.size() - self._capacity
            def _pop_over_cap():
                popped = self._queue.dequeue_up_to(over_cap)
                return tf.group(*snt.nest.flatten(popped))
            discard_op = tf.cond(over_cap <= 0, tf.no_op, _pop_over_cap)
        with tf.control_dependencies([discard_op]):
            replay_data = self._queue.dequeue_up_to(self._queue.size())
            reenqueue_op = self._queue.enqueue_many(replay_data)
        with tf.control_dependencies([reenqueue_op]):
            choose_idxs = tf.random_uniform(
                [batch_size],
                maxval=tf.shape(replay_data[0])[0],
                dtype=tf.int32)
            replay_serialized = snt.nest.pack_sequence_as(
                structure=self._types,
                flat_sequence=[
                    tf.gather(serialized_component, choose_idxs)
                    for serialized_component in replay_data[1:]])
            replay_sparse = snt.nest.map(tf.deserialize_many_sparse,
                                         replay_serialized, self._types)
            replay = snt.nest.map(tf.sparse_tensor_to_dense, replay_sparse)
            util.set_tensor_shapes(replay, self._sizes, add_batch_dims=2)
            return replay


def _scalar_summary(debug_tensors, name, tensor):
    """Add a summary and a debug output tensor."""
    tensor = tf.convert_to_tensor(tensor, name=name)
    debug_tensors[name] = tensor
    tf.summary.scalar(name, tensor)


def _dense_to_sparse(tensor):
    """Represent a dense tensor as a sparse one."""
    return tf.SparseTensor(
        indices=tf.where(tf.ones_like(tensor, dtype=tf.bool)),
        values=tf.reshape(tensor, [-1]),
        dense_shape=tf.shape(tensor, out_type=tf.int64))
