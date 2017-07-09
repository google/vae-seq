import tensorflow as tf
from tensorflow.contrib import distributions

class IndependentSequence(distributions.Distribution):
    """Wrapper distribution consisting of a sequence of independent events."""

    def __init__(self, item_dist, name=None):
        name = (name or self.__class__.__name__) + item_dist.name
        super(IndependentSequence, self).__init__(
            dtype=item_dist.dtype,
            reparameterization_type=item_dist.reparameterization_type,
            validate_args=item_dist.validate_args,
            allow_nan_stats=item_dist.allow_nan_stats,
            name=name)
        self._item_dist = item_dist

    def _batch_shape(self):
        return self._item_dist.batch_shape[:-1]

    def _batch_shape_tensor(self):
        return self._item_dist.batch_shape_tensor()[:-1]

    def _event_shape(self):
        return (self._item_dist.batch_shape[-1:]
                .concatenate(self._item_dist.event_shape))

    def _event_shape_tensor(self):
        return tf.concat([self._item_dist.batch_shape_tensor()[-1:],
                          self._item_dist.event_shape_tensor()], axis=0)

    def _log_prob(self, x):
        return tf.reduce_sum(self._item_dist.log_prob(x), axis=-1)

    def _prob(self, x):
        return tf.reduce_prod(self._item_dist.prob(x), axis=-1)

    def sample(self, *args, **kwargs):
        return self._item_dist.sample(*args, **kwargs)


@distributions.RegisterKL(IndependentSequence, IndependentSequence)
def _kl_independent_seq(dist_a, dist_b, name=None):
    name = name or 'KL_independent_seqs'
    with tf.name_scope(name):
        item_kl = distributions.kl_divergence(
            dist_a._item_dist, dist_b._item_dist)
        return tf.reduce_sum(item_kl, axis=-1)
