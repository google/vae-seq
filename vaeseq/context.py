"""Context modules summarize inputs and previous observations."""

import abc
import sonnet as snt
import tensorflow as tf

from . import util


def as_context(context, name=None):
    """Takes Tensors | Context and returns a Context."""
    if context is None:
        raise ValueError("Please supply a Context or a set of nested tensors.")
    if isinstance(context, Context):
        return context
    return Constant(context, name=name)


def as_tensors(context, observed):
    """Takes Tensors | Context and returns Tensors."""
    if context is None:
        raise ValueError("Please supply a Context or a set of nested tensors.")
    if isinstance(context, Context):
        context = context.from_observations(observed)
    return context


def _from_observations_cache_key(observations, initial_state):
    """Cache key used to memoize repeated calls to Context.from_observations."""
    flat_obs = snt.nest.flatten(observations)
    obs_names = tuple([obs.name for obs in flat_obs])
    state_names = None
    if initial_state is not None:
        flat_state = snt.nest.flatten(initial_state)
        state_names = tuple([st.name for st in flat_state])
    return (obs_names, state_names)


class Context(snt.RNNCore):
    """Context interface."""

    def __init__(self, name=None):
        super(Context, self).__init__(name=name)
        self._from_observations_cache = {}

    @abc.abstractproperty
    def output_size(self):
        """The non-batch sizes of the context Tensors."""

    @property
    def output_dtype(self):
        """The context Tensor types."""
        return snt.nest.map(lambda _: tf.float32, self.output_size)

    @abc.abstractproperty
    def state_size(self):
        """The non-batch sizes of this module's state Tensors."""

    @abc.abstractproperty
    def state_dtype(self):
        """The types of this module's state Tensors."""

    def initial_state(self, batch_size):
        def _zero_state(size, dtype):
            return tf.zeros([batch_size] + tf.TensorShape(size).as_list(),
                            dtype=dtype)
        return snt.nest.map(_zero_state, self.state_size, self.state_dtype)

    @abc.abstractmethod
    def _build(self, input_, state):
        """Returns a context for the current time step."""

    @abc.abstractmethod
    def observe(self, observation, state):
        """Returns the updated state."""

    def finished(self, state):
        """Returns whether each sequence in the batch has completed."""
        return False

    def drive_rnn(self,
                  cell,
                  sequence_size,
                  initial_state,
                  cell_initial_state,
                  cell_output_dtype=None,
                  cell_output_observations=lambda out: out):
        """Equivalent to tf.nn.dynamic_rnn, with inputs from this Context."""
        if cell_output_dtype is None:
            cell_output_dtype = snt.nest.map(
                lambda _: tf.float32, cell.output_size)
        def _loop_fn(time, cell_output, cell_state, ctx_state):
            if cell_state is None:
                cell_state = cell_initial_state
            if ctx_state is None:
                ctx_state = initial_state
            if cell_output is not None:
                obs = cell_output_observations(cell_output)
                ctx_state = self.observe(obs, ctx_state)
            finished = tf.logical_or(time >= sequence_size,
                                     self.finished(ctx_state))
            ctx, ctx_state = self(None, ctx_state)
            if cell_output is None:
                # tf.nn.raw_rnn uses the first cell_output as a dummy
                # to determine the output types and shapes. We need to
                # specify this to use heterogeneous output dtypes.
                # Note that the tensors here do not include the batch
                # dimension.
                with tf.name_scope("dummy"):
                    cell_output = snt.nest.map(
                        tf.zeros,
                        cell.output_size,
                        cell_output_dtype)
            return (finished, ctx, cell_state, cell_output, ctx_state)
        output_tas = tf.nn.raw_rnn(cell, _loop_fn)[0]
        outputs = snt.nest.map(
            lambda ta: util.transpose_time_batch(ta.stack()),
            output_tas)
        util.set_tensor_shapes(outputs, cell.output_size, add_batch_dims=2)
        return outputs

    def from_observations(self, observed, initial_state=None):
        """Generate contexts for a static sequence of observations."""
        cache_key = _from_observations_cache_key(observed, initial_state)
        if cache_key in self._from_observations_cache:
            return self._from_observations_cache[cache_key]
        with self._enter_variable_scope():
            with tf.name_scope("from_observations"):
                batch_size = util.batch_size_from_nested_tensors(observed)
                if initial_state is None:
                    initial_state = self.initial_state(batch_size)
                def _step(obs, state):
                    ctx, state = self(None, state)
                    state = self.observe(obs, state)
                    return ctx, state
                cell = util.WrapRNNCore(
                    _step,
                    state_size=self.state_size,
                    output_size=self.output_size)
                cell, observed = util.add_support_for_scalar_rnn_inputs(
                    cell, observed)
                contexts, _ = util.heterogeneous_dynamic_rnn(
                    cell, observed,
                    initial_state=initial_state,
                    output_dtypes=self.output_dtype)
                self._from_observations_cache[cache_key] = contexts
                return contexts


class Constant(Context):
    """Constant context wrapping a nested tuple of tensors."""

    def __init__(self, tensors, name=None):
        super(Constant, self).__init__(name=name)
        self._batch_size = util.batch_size_from_nested_tensors(tensors)
        self._sequence_size = util.sequence_size_from_nested_tensors(tensors)
        self._tensors = tensors

    @property
    def output_size(self):
        return snt.nest.map(lambda tensor: tensor.get_shape()[2:],
                            self._tensors)

    @property
    def output_dtype(self):
        return snt.nest.map(lambda tensor: tensor.dtype, self._tensors)

    @property
    def state_size(self):
        return tf.TensorShape([])

    @property
    def state_dtype(self):
        return tf.int32

    def initial_state(self, batch_size):
        del batch_size  # Ignore the requested batch size.
        return super(Constant, self).initial_state(self._batch_size)

    def observe(self, observation, state):
        del observation  # Not used.
        return state

    def finished(self, state):
        return state >= self._sequence_size

    def _build(self, input_, state):
        if input_ is not None:
            raise ValueError("I don't know how to encode any inputs.")
        finished = self.finished(state)
        state = tf.minimum(state, self._sequence_size - 1)
        indices = tf.concat([tf.expand_dims(tf.range(tf.shape(state)[0]), 1),
                             tf.expand_dims(state, 1)], axis=1)
        outputs = snt.nest.map(lambda tensor: tf.gather_nd(tensor, indices),
                               self._tensors)
        util.set_tensor_shapes(outputs, self.output_size, add_batch_dims=1)
        zero_outputs = snt.nest.map(tf.zeros_like, outputs)
        outputs = snt.nest.map(lambda zero, out: tf.where(finished, zero, out),
                               zero_outputs, outputs)
        return outputs, state + 1


class Chain(Context):
    """Compose a list of contexts."""

    def __init__(self, contexts, name=None):
        super(Chain, self).__init__(name=name)
        self._contexts = contexts

    @property
    def output_size(self):
        return self._contexts[-1].output_size

    @property
    def output_dtype(self):
        return self._contexts[-1].output_dtype

    @property
    def state_size(self):
        return tuple([ctx.state_size for ctx in self._contexts])

    @property
    def state_dtype(self):
        return tuple([ctx.state_dtype for ctx in self._contexts])

    def initial_state(self, batch_size):
        return [ctx.initial_state(batch_size) for ctx in self._contexts]

    def observe(self, observation, state):
        ret = []
        for context, ctx_state in zip(self._contexts, state):
            ret.append(context.observe(observation, ctx_state))
        return ret

    def finished(self, state):
        finished = False
        for context, ctx_state in zip(self._contexts, state):
            finished = tf.logical_or(finished, context.finished(ctx_state))
        return finished

    def _build(self, input_, state):
        ctx_out = input_
        ctx_states = []
        for context, ctx_state in zip(self._contexts, state):
            ctx_out, ctx_state = context(ctx_out, ctx_state)
            ctx_states.append(ctx_state)
        return ctx_out, ctx_states


class EncodeObserved(Context):
    """Simple context that encodes the input and previous observation."""

    def __init__(self, obs_encoder, input_encoder=None, name=None):
        super(EncodeObserved, self).__init__(name=name)
        self._input_encoder = input_encoder
        self._obs_encoder = obs_encoder

    @property
    def output_size(self):
        if self._input_encoder is None:
            return self._obs_encoder.output_size
        return (self._input_encoder.output_size,
                self._obs_encoder.output_size)

    @property
    def state_size(self):
        return self._obs_encoder.output_size

    @property
    def state_dtype(self):
        return tf.float32

    def observe(self, observation, state):
        del state  # Not used.
        return self._obs_encoder(observation)

    def _build(self, input_, state):
        if input_ is not None and self._input_encoder is None:
            raise ValueError("I don't know how to encode any inputs.")
        if self._input_encoder is None:
            ret = state
        else:
            ret = (self._input_encoder(input_), state)
        return ret, state


class Accumulate(Context):
    """Accumulates the last N observation encodings."""

    def __init__(self, obs_encoder, history_size, history_combiner, name=None):
        super(Accumulate, self).__init__(name=name)
        self._obs_encoder = obs_encoder
        self._history_size = history_size
        self._history_combiner = history_combiner

    @property
    def output_size(self):
        return self._history_combiner.output_size

    @property
    def state_size(self):
        obs_size = self._obs_encoder.output_size
        history_size = tf.TensorShape([self._history_size])
        return snt.nest.map(lambda size: history_size.concatenate(size),
                            obs_size)

    @property
    def state_dtype(self):
        return snt.nest.map(lambda _: tf.float32, self.state_size)

    def observe(self, observation, state):
        enc_obs = tf.expand_dims(self._obs_encoder(observation), axis=1)
        return snt.nest.map(
            lambda hist, obs: tf.concat([hist[:, 1:, :], obs], axis=1),
            state, enc_obs)

    def _build(self, input_, state):
        if input_ is not None:
            raise ValueError("I don't know how to encode any inputs.")
        return self._history_combiner(state), state
