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

"""Model base class used by the examples."""

from __future__ import print_function
from builtins import range

import abc
import six
import tensorflow as tf

from google.protobuf import text_format

from . import context as context_mod
from . import train as train_mod
from . import util
from . import vae as vae_mod


@six.add_metaclass(abc.ABCMeta)
class ModelBase(object):
    """Common functionality for training/generation/evaluation/etc."""

    def __init__(self, hparams, session_params):
        self._hparams = hparams
        self._session_params = session_params
        with tf.name_scope("model") as ns:
            self._name_scope = ns

    class SessionParams(object):
        """Utility class for commonly used session parameters."""

        def __init__(self, log_dir=None, master="", task=0,
                     session_config=None):
            self.log_dir = log_dir
            self.master = master
            self.task = task
            self.session_config = session_config

        @classmethod
        def add_parser_arguments(cls, parser):
            """Add ArgParse argument parsers for session flags.

            The result of argument parsing can be passed into the
            ModelBase constructor instead of a SessionParams object.
            """
            defaults = cls()
            parser.add_argument("--log-dir", required=True,
                                help="Checkpoint directory")
            parser.add_argument("--master", default=defaults.master,
                                help="Session master.")
            parser.add_argument("--task", default=defaults.task,
                                help="Worker task number.")
            def _parse_config_proto(msg):
                return text_format.Parse(msg, tf.ConfigProto())
            parser.add_argument("--session-config", default=None,
                                type=_parse_config_proto,
                                help="Session ConfigProto.")

    @property
    def hparams(self):
        return self._hparams

    def name_scope(self, name, default_name=None, values=None):
        with tf.name_scope(self._name_scope):
            # Capture the sub-namescope as an absolute path.
            with tf.name_scope(name, default_name, values) as ns:
                return tf.name_scope(ns)

    @util.lazy_property
    def encoder(self):
        with self.name_scope("encoder"):
            return self._make_encoder()

    @util.lazy_property
    def decoder(self):
        with self.name_scope("decoder"):
            return self._make_decoder()

    @util.lazy_property
    def feedback(self):
        with self.name_scope("feedback"):
            return self._make_feedback()

    @util.lazy_property
    def agent(self):
        with self.name_scope("agent"):
            return self._make_agent()

    @util.lazy_property
    def inputs(self):
        with self.name_scope("inputs"):
            return self._make_full_input_context(self.agent)

    @util.lazy_property
    def trainer(self):
        with self.name_scope("trainer"):
            return self._make_trainer()

    @util.lazy_property
    def vae(self):
        with self.name_scope("vae"):
            return vae_mod.make(self.hparams, self.encoder, self.decoder)

    def dataset(self, dataset, name=None):
        """Returns inputs and observations for the given dataset."""
        with self.name_scope("dataset", name):
            inputs, observed = self._make_dataset(dataset)
            return self._make_full_input_context(inputs), observed

    def training_session(self, hooks=None):
        scaffold = self._make_scaffold()
        return tf.train.MonitoredTrainingSession(
            master=self._session_params.master,
            config=self._session_params.session_config,
            is_chief=(self._session_params.task == 0),
            scaffold=scaffold,
            hooks=hooks,
            checkpoint_dir=self._session_params.log_dir)

    def eval_session(self, hooks=None):
        scaffold = self._make_scaffold()
        if self._session_params.task == 0:
            session_creator = tf.train.ChiefSessionCreator(
                master=self._session_params.master,
                config=self._session_params.session_config,
                scaffold=scaffold,
                checkpoint_dir=self._session_params.log_dir)
        else:
            session_creator = tf.train.WorkerSessionCreator(
                master=self._session_params.master,
                config=self._session_params.session_config,
                scaffold=scaffold)
        return tf.train.MonitoredSession(
            hooks=hooks,
            session_creator=session_creator)

    def evaluate(self, dataset, num_steps):
        """Calculates the mean log-prob for the given sequences."""
        with tf.name_scope("evaluate"):
            inputs, observed = self.dataset(dataset)
            log_probs = self.vae.evaluate(
                inputs, observed,
                samples=self.hparams.log_prob_samples)
            mean_log_prob, update = tf.metrics.mean(log_probs)
        hooks = [tf.train.LoggingTensorHook({"log_prob": mean_log_prob},
                                            every_n_secs=10., at_end=True)]
        latest = None
        with self.eval_session(hooks=hooks) as sess:
            for _ in range(num_steps):
                if sess.should_stop():
                    break
                latest = sess.run(update)
        if latest is not None:
            return {"log_prob": latest}
        return None

    def train(self, dataset, num_steps, valid_dataset=None):
        """Trains/continues training the model."""
        global_step = tf.train.get_or_create_global_step()
        inputs, observed = self.dataset(dataset, name="train_dataset")
        train_op, debug = self.trainer(inputs, observed)
        debug["global_step"] = global_step
        hooks = [tf.train.LoggingTensorHook(debug, every_n_secs=60.)]
        if self._session_params.log_dir:
            # Add metric summaries to be computed at a slower rate.
            slow_summaries = []
            def _add_to_slow_summaries(name, inputs, observed):
                """Creates a self-updating metric summary op."""
                with tf.name_scope(name):
                    log_probs = self.vae.evaluate(
                        inputs, observed,
                        samples=self.hparams.log_prob_samples)
                    mean, update = tf.metrics.mean(log_probs)
                    with tf.control_dependencies([update]):
                        slow_summaries.append(
                            tf.summary.scalar("mean_log_prob",
                                              mean, collections=[]))
            _add_to_slow_summaries("train_eval", inputs, observed)
            if valid_dataset is not None:
                vinputs, vobserved = self.dataset(valid_dataset,
                                                  name="valid_dataset")
                _add_to_slow_summaries("valid_eval", vinputs, vobserved)
            hooks.append(tf.train.SummarySaverHook(
                save_steps=100,
                output_dir=self._session_params.log_dir,
                summary_op=tf.summary.merge(slow_summaries)))

            # Add sample generated sequences.
            generated, _unused_latents = self.vae.generate(
                inputs=inputs,
                batch_size=util.batch_size_from_nested_tensors(observed),
                sequence_size=util.sequence_size_from_nested_tensors(observed))
            hooks.append(tf.train.SummarySaverHook(
                save_steps=1000,
                output_dir=self._session_params.log_dir,
                summary_op=tf.summary.merge([
                    self._make_output_summary("observed", observed),
                    self._make_output_summary("generated", generated),
                ])))

        debug_vals = None
        with self.training_session(hooks=hooks) as sess:
            for local_step in range(num_steps):
                if sess.should_stop():
                    break
                if local_step < num_steps - 1:
                    sess.run(train_op)
                else:
                    _, debug_vals = sess.run((train_op, debug))
        return debug_vals

    def generate(self):
        """Generates sequences from a trained model."""
        generated, _unused_latents = self.vae.generate(self.inputs)
        rendered = self._render(generated)
        with self.eval_session() as sess:
            while True:
                batch = sess.run(rendered)
                for sequence in batch:
                    yield sequence

    def _make_full_input_context(self, inputs):
        """Chains agent with feedback produce the VAE input."""
        if inputs is None:
            return self.feedback
        inputs = context_mod.as_context(inputs)
        return context_mod.Chain([inputs, self.feedback])

    def _make_scaffold(self):
        local_init_op = tf.group(
            tf.local_variables_initializer(),
            tf.tables_initializer(),
            *tf.get_collection(tf.GraphKeys.LOCAL_INIT_OP))
        return tf.train.Scaffold(local_init_op=local_init_op)

    def _render(self, observed):
        """Returns a rendering of the modeled observation for output."""
        return observed

    def _make_feedback(self):
        """Constructs the feedback Context."""
        # Default to an encoding of the previous observation..
        return context_mod.EncodeObserved(self.encoder)

    def _make_agent(self):
        """Constructs a Context used for generating inputs."""
        return None  # No inputs.

    def _make_trainer(self):
        global_step = tf.train.get_or_create_global_step()
        loss = train_mod.ELBOLoss(self.hparams, self.vae)
        return train_mod.Trainer(self.hparams, global_step=global_step,
                                 loss=loss, variables=tf.trainable_variables)

    @abc.abstractmethod
    def _make_encoder(self):
        """Constructs the observation encoder."""

    @abc.abstractmethod
    def _make_decoder(self):
        """Constructs the observation decoder DistModule."""

    @abc.abstractmethod
    def _make_dataset(self, dataset):
        """Returns inputs (can be None) and outputs as sequence Tensors."""

    @abc.abstractmethod
    def _make_output_summary(self, tag, observed):
        """Returns a tf.summary to display this sequence.."""
