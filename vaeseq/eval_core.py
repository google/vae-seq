"""Module for evaluating the likelihood of a given sequence."""

import tensorflow as tf
import sonnet as snt

from . import agent as agent_mod
from . import dist_module
from . import util


class EvalCore(snt.RNNCore):
    """An RNNCore that outputs log-probabilities for the given observations.

    Note that the log-probabilities will be based on sampled latent states.
    """

    def __init__(self, agent, latent_distcore, obs_distcore, name=None):
        super(EvalCore, self).__init__(name=name)
        assert isinstance(agent, agent_mod.Agent)
        self._agent = agent
        with self._enter_variable_scope():
            self._from_contexts = EvalCoreFromContexts(latent_distcore,
                                                       obs_distcore)

    @property
    def agent(self):
        return self._agent

    @property
    def from_contexts(self):
        """Return an EvalCore that operates on Agent contexts."""
        return self._from_contexts

    @property
    def state_size(self):
        """Sizes of state tensors."""
        return (self.agent.state_size, self.from_contexts.state_size)

    def initial_state(self, batch_size):
        """Override default implementation to support heterogeneous dtypes."""
        # TODO: support trainable initial states.
        return (self.agent.initial_state(batch_size),
                self.from_contexts.initial_state(batch_size))

    @property
    def output_size(self):
        """Sizes of output tensors."""
        return self.from_contexts.output_size

    def _build(self, input_obs, state):
        input_, obs = input_obs
        agent_state, inner_state = state
        context = self.agent.context(input_, agent_state)
        output, inner_state = self.from_contexts((context, obs), inner_state)
        agent_state = self.agent.observe(input_, obs, agent_state)
        state = (agent_state, inner_state)
        return output, state

    def log_probs(self, agent_inputs, observed, initial_state=None, samples=1):
        """Compute a monte-carlo estimate of log-prob(observed)."""
        if initial_state is None:
            batch_size = util.batch_size_from_nested_tensors(observed)
            initial_state = self.initial_state(batch_size)
        cell = self
        inputs = (agent_inputs, observed)
        cell, inputs = util.add_support_for_scalar_rnn_inputs(cell, inputs)
        return _average_runs(samples, cell, inputs, initial_state)


class EvalCoreFromContexts(snt.RNNCore):
    """Same as EvalCore, but start from contexts rather than agent inputs."""

    def __init__(self, latent_distcore, obs_distcore, name=None):
        super(EvalCoreFromContexts, self).__init__(name=name)
        assert isinstance(latent_distcore, dist_module.DistCore)
        assert isinstance(obs_distcore, dist_module.DistCore)
        self._latent_distcore = latent_distcore
        self._obs_distcore = obs_distcore

    @property
    def state_size(self):
        """Sizes of state tensors."""
        return (self._latent_distcore.state_size,  # latent core state
                self._obs_distcore.state_size,)    # observation core state

    def initial_state(self, batch_size):
        """Override default implementation to support heterogeneous dtypes."""
        # TODO: support trainable initial states.
        return (self._latent_distcore.samples.initial_state(batch_size),
                self._obs_distcore.samples.initial_state(batch_size),)

    @property
    def output_size(self):
        """Sizes of output tensors."""
        return tf.TensorShape([])  # log_prob

    def _build(self, context_obs, state):
        context, obs = context_obs
        latent_state, obs_state = state
        latent, latent_state = self._latent_distcore.samples(
            context, latent_state)
        output, obs_state = self._obs_distcore.log_probs(
            ((context, latent), obs), obs_state)
        state = (latent_state, obs_state)
        return output, state

    def log_probs(self, contexts, observed, initial_state=None, samples=1):
        """Compute a monte-carlo estimate of log-prob(observed)."""
        if initial_state is None:
            batch_size = util.batch_size_from_nested_tensors(observed)
            initial_state = self.initial_state(batch_size)
        cell = self
        inputs = (contexts, observed)
        cell, inputs = util.add_support_for_scalar_rnn_inputs(cell, inputs)
        return _average_runs(samples, cell, inputs, initial_state)


def _average_runs(num_runs, cell, inputs, initial_state):
    """Run the RNN outputs over num_run runs."""
    def _run(unused_arg):
        del unused_arg
        return tf.nn.dynamic_rnn(
            cell, inputs,
            initial_state=initial_state,
            dtype=tf.float32)[0]
    if num_runs == 1:
        return _run(None)
    runs = tf.map_fn(_run, tf.zeros([num_runs, 0]))
    return tf.reduce_mean(runs, axis=0)
