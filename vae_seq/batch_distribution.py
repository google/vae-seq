"""A distribution over (independent) batches of events."""

import tensorflow as tf


class BatchDistribution(tf.distributions.Distribution):
    """Wrap a distribution to shift batch dimensions into the event shape."""

    def __init__(self, distribution, ndims=1, name=None):
        parameters = locals()
        self._dist = distribution
        self._ndims = ndims
        super(BatchDistribution, self).__init__(
            dtype=distribution.dtype,
            reparameterization_type=distribution.reparameterization_type,
            validate_args=distribution.validate_args,
            allow_nan_stats=distribution.allow_nan_stats,
            parameters=parameters,
            graph_parents=distribution._graph_parents,
            name=name or "Batch" + distribution.name
        )

    def _sample_n(self, n, seed=None):
        return self._dist._sample_n(n, seed=seed)

    def _batch_shape_tensor(self):
        return self._dist.batch_shape_tensor()[:-self._ndims]

    def _batch_shape(self):
        return self._dist.batch_shape[:-self._ndims]

    def _event_shape_tensor(self):
        batch_dims = self._dist.batch_shape_tensor()[-self._ndims:]
        return tf.concat([batch_dims, self._dist.event_shape_tensor()], 0)

    def _event_shape(self):
        batch_dims = self._dist.batch_shape[-self._ndims:]
        return batch_dims.concatenate(self._dist.event_shape)

    def _log_prob(self, event):
        log_probs = self._dist._log_prob(event)
        return tf.reduce_sum(log_probs, axis=list(range(-self._ndims, 0)))

    def _prob(self, event):
        probs = self._dist._prob(event)
        return tf.reduce_prod(probs, axis=list(range(-self._ndims, 0)))
