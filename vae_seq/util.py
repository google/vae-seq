"""Utilities used elsewhere in this library."""

import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions


def calc_kl(hparams, a_sample, dist_a, dist_b):
    """Calculates KL(a||b), either analytically or via MC estimate."""
    if hparams.use_monte_carlo_kl:
        return dist_a.log_prob(a_sample) - dist_b.log_prob(a_sample)
    return distributions.kl_divergence(dist_a, dist_b)


def activation(hparams):
    """Returns the activation function selected in hparams."""
    return {
        "relu": tf.nn.relu,
        "elu": tf.nn.elu,
    }[hparams.activation]


def positive_projection(hparams):
    """Returns the positive projection selected in hparams."""
    return {
        "exp": tf.exp,
        "softplus": tf.nn.softplus,
    }[hparams.positive_projection]


def make_rnn(hparams, name):
    """Constructs a DeepRNN using hparams.rnn_hidden_sizes."""
    with tf.variable_scope(name):
        return snt.DeepRNN(
            [snt.LSTM(size) for size in hparams.rnn_hidden_sizes], name=name)


def make_mlp(hparams, layers, name=None):
    """Constructs an MLP with the given layers, using hparams.activation."""
    return snt.nets.MLP(
        layers, activation=activation(hparams), name=name or "MLP")


def concat_features(tensors):
    """Concatenates nested tensors along the last dimension."""
    tensors = snt.nest.flatten(tensors)
    if len(tensors) == 1:
        return tensors[0]
    return tf.concat(tensors, axis=-1)


class WrapRNNCore(snt.RNNCore):
    """Wrap a transition function into an RNNCore."""

    def __init__(self, step, state_size, output_size, name=None):
        super(WrapRNNCore, self).__init__(name=name)
        self._step = step
        self._state_size = state_size
        self._output_size = output_size

    @property
    def output_size(self):
        """RNN output sizes."""
        return self._output_size

    @property
    def state_size(self):
        """RNN state sizes."""
        return self._state_size

    def _build(self, input_, state):
        return self._step(input_, state)


def heterogeneous_dynamic_rnn(
        cell, inputs, initial_state=None, time_major=False,
        output_dtypes=None, **kwargs):
    """Wrapper around tf.nn.dynamic_rnn that supports heterogeneous outputs."""
    time_axis = 0 if time_major else 1
    batch_axis = 1 if time_major else 0
    first_input_shape = tf.shape(snt.nest.flatten(inputs)[0])
    if initial_state is None:
        batch_size = first_input_shape[batch_axis]
        initial_state = cell.zero_state(batch_size, output_dtypes)
    flat_dtypes = snt.nest.flatten(output_dtypes)
    flat_output_size = snt.nest.flatten(cell.output_size)
    # The first output will be returned the normal way; the rest will
    # be returned via state TensorArrays.
    input_length = first_input_shape[time_axis]
    aux_output_tas = [
        tf.TensorArray(
            dtype,
            size=input_length,
            element_shape=tf.TensorShape([None]).concatenate(out_size))
        for dtype, out_size in zip(flat_dtypes, flat_output_size)[1:]
    ]
    aux_state = (0, aux_output_tas, initial_state)

    def _step(input_, (step, aux_output_tas, state)):
        """Wrap the cell to return the first output and store the rest."""
        outputs, state = cell(input_, state)
        flat_outputs = snt.nest.flatten(outputs)
        aux_output_tas = [
            ta.write(step, output)
            for ta, output in zip(aux_output_tas, flat_outputs[1:])
        ]
        return flat_outputs[0], (step + 1, aux_output_tas, state)

    first_output, (_, aux_output_tas, state) = tf.nn.dynamic_rnn(
        WrapRNNCore(_step, state_size=None, output_size=flat_output_size[0]),
        inputs,
        initial_state=aux_state,
        dtype=flat_dtypes[0],
        time_major=time_major,
        **kwargs)
    first_output_shape = first_output.get_shape().with_rank_at_least(2)
    time_and_batch = tf.TensorShape([first_output_shape[time_axis],
                                     first_output_shape[batch_axis]])
    outputs = [first_output]
    for aux_output_ta in aux_output_tas:
        output = aux_output_ta.stack()
        output.set_shape(time_and_batch.concatenate(output.get_shape()[2:]))
        if not time_major:
            output = transpose_time_batch(output)
        outputs.append(output)
    return snt.nest.pack_sequence_as(output_dtypes, outputs), state


def transpose_time_batch(tensor):
    """Transposes the first two dimensions of a Tensor."""
    perm = range(tensor.get_shape().with_rank_at_least(2).ndims)
    perm[0], perm[1] = 1, 0
    return tf.transpose(tensor, perm=perm)


def reverse_dynamic_rnn(cell, inputs, time_major=False, **kwargs):
    """Runs tf.nn.dynamic_rnn backwards."""
    time_axis = 0 if time_major else 1
    reverse_seq = lambda x: tf.reverse(x, axis=[time_axis])
    inputs = snt.nest.map(reverse_seq, inputs)
    output, state = tf.nn.dynamic_rnn(
        cell, inputs, time_major=time_major, **kwargs)
    return snt.nest.map(reverse_seq, output), state


def dynamic_hparam(key, value):
    """Returns a memoized, non-constant Tensor that allows feeding."""
    collection = tf.get_collection_ref("HPARAMS_" + key)
    if len(collection) > 1:
        raise ValueError("Dynamic hparams ollection should contain one item.")
    if not collection:
        with tf.name_scope(""):
            default_value = tf.convert_to_tensor(value, name=key + "_default")
            tensor = tf.placeholder_with_default(
                default_value,
                default_value.get_shape(),
                name=key)
            collection.append(tensor)
    return collection[0]


def batch_size(hparams):
    """Returns a non-constant Tensor that evaluates to hparams.batch_size."""
    return dynamic_hparam("batch_size", hparams.batch_size)


def sequence_size(hparams):
    """Returns a non-constant Tensor that evaluates to hparams.sequence_size."""
    return dynamic_hparam("sequence_size", hparams.sequence_size)
