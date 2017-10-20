"""Base classes for modules that return distributions."""

import abc
import functools
import tensorflow as tf
import sonnet as snt

from . import util


class DistModule(snt.AbstractModule):
    """A module that returns parameters for a Distribution."""

    @abc.abstractproperty
    def event_dtype(self):
        """Returns the output distribution event dtypes."""

    @abc.abstractproperty
    def event_size(self):
        """Returns the output distribution event sizes."""

    @abc.abstractmethod
    def dist(self, params, name=None):
        """Constructs a Distribution parameterized by the module output."""


class DistCore(DistModule):
    """Like an RNNCore, but outputs distributions."""

    @abc.abstractproperty
    def state_size(self):
        """Returns the non-batched sizes of Tensors returned from next_state."""

    @abc.abstractmethod
    def _next_state(self, state_arg, event=None):
        """Produces the next state given a state_arg and event.
        NOTE: this function shouldn't allocate variables."""

    def next_sample(self, input_, state, with_log_prob=False):
        """Returns the next sample and state from the distribution."""
        dist_arg, state_arg = self(input_, state)
        dist = self.dist(dist_arg)
        event = dist.sample()
        util.set_tensor_shapes(event, dist.event_shape, add_batch_dims=1)
        state = self._next_state(state_arg, event)
        if with_log_prob:
            return (event, dist.log_prob(event)), state
        return event, state

    def next_log_prob(self, input_and_observed, state):
        """Returns the log-prob(observed) for the next step."""
        input_, observed = input_and_observed
        dist_arg, state_arg = self(input_, state)
        dist = self.dist(dist_arg)
        state = self._next_state(state_arg, observed)
        return dist.log_prob(observed), state

    @util.lazy_property
    def samples(self):
        """Returns an RNNCore that produces a sequence of samples."""
        return util.WrapRNNCore(
            functools.partial(self.next_sample, with_log_prob=False),
            self.state_size,
            output_size=self.event_size,
            name=self.module_name + "/samples")


    @util.lazy_property
    def samples_with_log_probs(self):
        """Returns an RNNCore that produces (sample, log-prob(sample))."""
        return util.WrapRNNCore(
            functools.partial(self.next_sample, with_log_prob=True),
            self.state_size,
            output_size=(self.event_size, tf.TensorShape([])),
            name=self.module_name + "/samples_with_log_probs")

    @util.lazy_property
    def log_probs(self):
        """Returns an RNNCore that evaluates the log-prob of the input."""
        return util.WrapRNNCore(
            self.next_log_prob,
            self.state_size,
            output_size=tf.TensorShape([]),
            name=self.module_name + "/log_probs")
