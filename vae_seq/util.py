"""Utilities used elsewhere in this library."""

import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions


def calc_kl(hparams, q_sample, dist_q, dist_p):
    """Calculates a KL divergence, either analytically or via MC estimate."""
    if hparams.use_monte_carlo_kl:
        return dist_q.log_prob(q_sample) - dist_p.log_prob(q_sample)
    return distributions.kl_divergence(dist_q, dist_p)


def reverse_dynamic_rnn(cell, inputs, time_major=False, **kwargs):
    """Run tf.nn.dynamic_rnn backwards."""
    time_axis = 0 if time_major else 1
    reverse_seq = lambda x: tf.reverse(x, axis=[time_axis])
    inputs = snt.nest.map(reverse_seq, inputs)
    output, state = tf.nn.dynamic_rnn(
        cell, inputs, time_major=time_major, **kwargs)
    return snt.nest.map(reverse_seq, output), state


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
        super(WrapRNNCore, self).__init__(name or self.__class__.__name__)
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
