import collections
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions


VAETensors = collections.namedtuple('VAETensors', [
    'gen_z',         # gen_z ~ p(gen_z)
    'gen_x',         # gen_x ~ p(gen_x | gen_z)
    'gen_log_prob',  # p(gen_x = obs | z)
    'inf_z',         # inf_z ~ q(inf_z | obs)
    'inf_x',         # inf_x ~ p(inf_x | inf_z) -- just for debugging.
    'inf_log_prob',  # p(inf_x = obs | inf_z)
    'inf_kl',        # kl(q(inf_z | obs) || p(z))
])


def calc_kl(hparams, q_sample, dist_q, dist_p):
    if hparams.use_monte_carlo_kl:
        return dist_q.log_prob(q_sample) - dist_p.log_prob(q_sample)
    return distributions.kl_divergence(dist_q, dist_p)


def reverse_dynamic_rnn(cell, inputs, time_major=False, **kwargs):
    time_axis = 0 if time_major else 1
    reverse_seq = lambda x: tf.reverse(x, axis=[time_axis])
    inputs = snt.nest.map(reverse_seq, inputs)
    output, state = tf.nn.dynamic_rnn(cell, inputs, time_major=time_major, **kwargs)
    return snt.nest.map(reverse_seq, output), state


def squeeze_sum(x):
    return tf.reduce_sum(tf.squeeze(x, axis=2), axis=1)


def activation(hparams):
    return {
        'relu': tf.nn.relu,
        'elu': tf.nn.elu,
    }[hparams.activation]


def positive_projection(hparams):
    return {
        'exp': tf.exp,
        'softplus': tf.nn.softplus,
    }[hparams.positive_projection]


def make_rnn(hparams, name=None):
    return snt.DeepRNN(
        [snt.LSTM(size) for size in hparams.rnn_hidden_sizes],
        name=name or 'RNN')


def make_mlp(hparams, layers, name=None):
    return snt.nets.MLP(
        layers, activation=activation(hparams), name=name or 'MLP')


def concat_features(xs):
    if len(xs) == 1:
        return xs[0]
    return tf.concat(xs, axis=-1)
