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

"""Training subgraph for a VAE."""

import sonnet as snt
import tensorflow as tf

from . import util


class TrainOps(snt.AbstractModule):
    """This module produces a train_op given Agent contexts and observations."""

    def __init__(self, hparams, vae, name=None):
        super(TrainOps, self).__init__(name=name)
        self._hparams = hparams
        self._vae = vae

    def _build(self, contexts, observed):
        hparams = self._hparams
        latents, divs = self._vae.infer_latents(contexts, observed)
        log_probs = self._vae.log_prob_observed(contexts, latents, observed)

        # Compute the ELBO.
        batch_size = tf.to_float(util.batch_size(hparams))
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
