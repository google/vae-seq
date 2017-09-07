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

"""Training subgraph for a VAE."""

import functools
import numpy as np
import sonnet as snt
import tensorflow as tf

from . import util


class Trainer(snt.AbstractModule):
    """Wraps an optimizer and an objective."""

    def __init__(self, hparams, global_step, loss,
                 variables=tf.trainable_variables,
                 name=None):
        super(Trainer, self).__init__(name=name)
        self._hparams = hparams
        self._global_step = global_step
        self._loss = loss
        if callable(variables):
            self._variables = variables
        else:
            self._variables = lambda: variables

    @util.lazy_property
    def optimizer(self):
        return self._make_optimizer()

    def _make_optimizer(self):
        return tf.train.AdamOptimizer(self._hparams.learning_rate)

    def _transform_gradients(self, gradients_to_variables):
        """Transform gradients before applying the optimizer."""
        if self._hparams.clip_gradient_norm > 0:
            gradients_to_variables = tf.contrib.training.clip_gradient_norms(
                gradients_to_variables,
                self._hparams.clip_gradient_norm)
        return gradients_to_variables

    def _build(self, inputs, observed):
        loss, debug_tensors = self._loss(inputs, observed)
        variables = self._variables()
        if not variables:
            raise ValueError("No trainable variables found.")
        # Summarize the magnitudes of the model variables to see whether we need
        # regularization.
        for var in variables:
            tf.summary.histogram(name=var.op.name + "/values", values=var)
        # Unfortunately regularization losses are all stored together
        # so we can't segment them by those that come from variables.
        reg_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar(name="reg_loss", tensor=reg_loss)
        train_op = tf.contrib.training.create_train_op(
            loss + reg_loss,
            self.optimizer,
            global_step=self._global_step,
            variables_to_train=variables,
            transform_grads_fn=self._transform_gradients,
            summarize_gradients=True,
            check_numerics=self._hparams.check_numerics)
        return train_op, debug_tensors


class Group(snt.AbstractModule):
    """Trainer that joins multiple trainers together."""

    def __init__(self, trainers, name=None):
        super(Group, self).__init__(name=name)
        self._trainers = trainers

    def _build(self, inputs, observed):
        train_ops = []
        debug_tensors = {}
        for trainer in self._trainers:
            train_op, debug = trainer(inputs, observed)
            train_ops.append(train_op)
            debug_tensors.update(debug)
        return tf.group(*train_ops), debug_tensors


class ELBOLoss(snt.AbstractModule):
    """Calculates an objective for maximizing the evidence lower bound."""

    def __init__(self, hparams, vae, name=None):
        super(ELBOLoss, self).__init__(name=name)
        self._hparams = hparams
        self._vae = vae

    def _build(self, inputs, observed):
        debug_tensors = {}
        scalar_summary = functools.partial(_scalar_summary, debug_tensors)

        latents, divs = self._vae.infer_latents(inputs, observed)
        log_probs = self._vae.evaluate(inputs, observed, latents=latents)
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
        scalar_summary(self.module_name, loss)
        return loss, debug_tensors


class RewardLoss(snt.AbstractModule):
    """Sums component losses."""

    def __init__(self, hparams, inputs, vae, reward, name=None):
        super(RewardLoss, self).__init__(name=name)
        self._inputs = inputs
        self._vae = vae
        self._reward = reward

    def _build(self, inputs, observed):
        del inputs, observed  # We only use the generated reward.
        debug_tensors = {}
        scalar_summary = functools.partial(_scalar_summary, debug_tensors)

        generated = self._vae.generate(self._inputs)[0]
        mean_reward = tf.reduce_mean(self._reward(generated))
        scalar_summary("mean_reward", mean_reward)
        #loss = -self._neg_log_reward_dist.log_prob(-tf.log(mean_reward + 1e-5))
        loss = -mean_reward
        scalar_summary(self.module_name, loss)
        return loss, debug_tensors


def _scalar_summary(debug_tensors, name, tensor):
    """Add a summary and a debug output tensor."""
    tensor = tf.convert_to_tensor(tensor, name=name)
    debug_tensors[name] = tensor
    tf.summary.scalar(name, tensor)
