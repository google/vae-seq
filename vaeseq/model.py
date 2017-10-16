"""Model base class used by the examples."""

from __future__ import print_function
from builtins import range

import abc
import six
import tensorflow as tf

from . import train as train_mod
from . import util


@six.add_metaclass(abc.ABCMeta)
class ModelBase(object):
    """Common functionality for training/generation/evaluation/etc."""

    def __init__(self, hparams, session_params):
        self._hparams = hparams
        self._session_params = session_params

    class SessionParams(object):
        """Utility class for commonly used session parameters."""

        def __init__(self, log_dir=None, master="", task=0):
            self.log_dir = log_dir
            self.master = master
            self.task = task

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


    @property
    def hparams(self):
        return self._hparams

    @util.lazy_property
    def vae(self):
        with tf.name_scope("vae"):
            return self._make_vae()

    @util.lazy_property
    def trainer(self):
        return train_mod.Trainer(self.hparams, self.vae,
                                 tf.train.get_or_create_global_step())

    def training_session(self, hooks=None):
        scaffold = self._make_scaffold()
        return tf.train.MonitoredTrainingSession(
            master=self._session_params.master,
            is_chief=(self._session_params.task == 0),
            scaffold=scaffold,
            hooks=hooks,
            checkpoint_dir=self._session_params.log_dir)

    def eval_session(self, hooks=None):
        scaffold = self._make_scaffold()
        if self._session_params.task == 0:
            session_creator = tf.train.ChiefSessionCreator(
                master=self._session_params.master,
                scaffold=scaffold,
                checkpoint_dir=self._session_params.log_dir)
        else:
            session_creator = tf.train.WorkerSessionCreator(
                master=self._session_params.master,
                scaffold=scaffold)
        return tf.train.MonitoredSession(
            hooks=hooks,
            session_creator=session_creator)

    def evaluate(self, dataset, num_steps):
        """Calculates the mean log-prob for the given sequences."""
        with tf.name_scope("evaluate"):
            contexts, observed = self._open_dataset(dataset)
            log_probs = self.vae.eval_core.from_contexts.log_probs(
                contexts, observed)
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
        with tf.name_scope("train"):
            contexts, observed = self._open_dataset(dataset)
        train_op, debug = self.trainer(contexts, observed)

        hooks = [tf.train.LoggingTensorHook(debug, every_n_secs=60.)]
        if self._session_params.log_dir:
            # Add metric summaries to be computed at a slower rate.
            slow_summaries = []
            def _add_to_slow_summaries(name, contexts, observed):
                """Creates a self-updating metric summary op."""
                with tf.name_scope(name):
                    log_probs = self.vae.eval_core.from_contexts.log_probs(
                        contexts, observed)
                    mean, update = tf.metrics.mean(log_probs)
                    with tf.control_dependencies([update]):
                        slow_summaries.append(
                            tf.summary.scalar("mean_log_prob",
                                              mean, collections=[]))
            _add_to_slow_summaries("train_eval", contexts, observed)
            if valid_dataset is not None:
                with tf.name_scope("valid"):
                    vcontexts, vobserved = self._open_dataset(valid_dataset)
                _add_to_slow_summaries("valid_eval", vcontexts, vobserved)
            hooks.append(tf.train.SummarySaverHook(
                save_steps=100,
                output_dir=self._session_params.log_dir,
                summary_op=tf.summary.merge(slow_summaries)))

            # Add sample generated sequences.
            batch_size = util.batch_size_from_nested_tensors(observed)
            sequence_size = util.sequence_size_from_nested_tensors(observed)
            generated = self.vae.gen_core.generate(
                self.vae.agent.get_inputs(batch_size, sequence_size))[0]
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
        generated = self.vae.gen_core.generate(
            self.vae.agent.get_inputs(util.batch_size(self.hparams),
                                      util.sequence_size(self.hparams)))[0]
        rendered = self._render(generated)
        with self.eval_session() as sess:
            batch = sess.run(rendered)
            for sequence in batch:
                yield sequence

    def _make_scaffold(self):
        local_init_op = tf.group(
            tf.local_variables_initializer(),
            tf.tables_initializer(),
            *tf.get_collection(tf.GraphKeys.LOCAL_INIT_OP))
        return tf.train.Scaffold(local_init_op=local_init_op)

    def _render(self, observed):
        """Returns a rendering of the modeled observation for output."""
        return observed

    def _make_elbo_loss(self):
        """Returns a (contexts, observed) -> (loss, debug_tensors) map."""
        return train_mod.ELBOLoss(self.hparams, self.vae)

    def _make_agent_loss(self):
        """Returns a (observed, log_probs) -> (loss, debug_tensors) map."""
        return None

    @abc.abstractmethod
    def _make_vae(self):
        """Constructs the VAE."""

    @abc.abstractmethod
    def _open_dataset(self, dataset):
        """Returns contexts and observations."""

    @abc.abstractmethod
    def _make_output_summary(self, tag, observed):
        """Returns a tf.summary to display this sequence.."""
