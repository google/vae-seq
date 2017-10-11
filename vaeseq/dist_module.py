"""Base classes for modules that return distributions."""

import abc
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

    @property
    def samples(self):
        """Returns an RNNCore that outputs samples."""
        def _step(input_, state):
            """Samples from the distribution at each time step."""
            dist_arg, state_arg = self(input_, state)
            dist = self.dist(dist_arg)
            event = dist.sample()
            util.set_tensor_shapes(event, dist.event_shape, add_batch_dim=True)
            state = self._next_state(state_arg, event)
            return event, state
        return util.WrapRNNCore(
            _step,
            self.state_size,
            output_size=self.event_size,
            name=self.module_name + "/samples")

    @property
    def log_probs(self):
        """Returns an RNNCore that outputs log-probabilities."""
        def _step(input_obs, state):
            """Calculates the log-probability of the observed event."""
            input_, observed = input_obs
            dist_arg, state_arg = self(input_, state)
            dist = self.dist(dist_arg)
            state = self._next_state(state_arg, observed)
            return dist.log_prob(observed), state
        return util.WrapRNNCore(
            _step,
            self.state_size,
            output_size=tf.TensorShape([]),
            name=self.module_name + "/log_probs")
