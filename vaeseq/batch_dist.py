# Copyright 2018 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributions over independent sets of events."""

import sonnet as snt
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
            name=name or "batch_" + distribution.name
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


class GroupDistribution(tf.distributions.Distribution):
    """Group together several independent distributions.

    Note, the batch shapes of the component distributions must match.
    """

    def __init__(self, distributions, name=None):
        parameters = locals()
        self._dists = distributions
        self._flat_dists = snt.nest.flatten(distributions)
        dtype = snt.nest.map(lambda dist: dist.dtype, distributions)
        r16n_type = tf.distributions.FULLY_REPARAMETERIZED
        for dist in self._flat_dists:
            r16n_type = dist.reparameterization_type
            if r16n_type is not tf.distributions.FULLY_REPARAMETERIZED:
                break
        validate_args = all([dist.validate_args for dist in self._flat_dists])
        allow_nan_stats = all(
            [dist.allow_nan_stats for dist in self._flat_dists])
        graph_parents = snt.nest.flatten(
            [dist._graph_parents for dist in self._flat_dists])
        name = name or "_".join([dist.name for dist in self._flat_dists])
        super(GroupDistribution, self).__init__(
            dtype=dtype,
            reparameterization_type=r16n_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=graph_parents,
            name=name)

    @property
    def batch_shape(self):
        return snt.nest.map(lambda dist: dist.batch_shape, self._dists)

    def batch_shape_tensor(self, name="batch_shape_tensor"):
        with self._name_scope(name):
            return snt.nest.map(
                lambda dist: dist.batch_shape_tensor(name), self._dists)

    @property
    def event_shape(self):
        return snt.nest.map(lambda dist: dist.event_shape, self._dists)

    def event_shape_tensor(self, name="event_shape_tensor"):
        with self._name_scope(name):
            return snt.nest.map(
                lambda dist: dist.event_shape_tensor(name), self._dists)

    def _is_scalar_helper(self, *args, **kwargs):
        if not self._flat_dists:
            return True
        if len(self._flat_dists) == 1:
            return self._flat_dists[0]._is_scalar_helper(*args, **kwargs)
        return False

    def sample(self, *args, **kwargs):
        return snt.nest.map(
            lambda dist: dist.sample(*args, **kwargs),
            self._dists)

    def log_prob(self, value, name="log_prob"):
        flat_values = snt.nest.flatten(value)
        with self._name_scope(name, values=flat_values):
            return tf.reduce_sum(
                [dist.log_prob(val)
                 for dist, val in zip(self._flat_dists, flat_values)],
                axis=0)

    def prob(self, value, name="prob"):
        flat_values = snt.nest.flatten(value)
        with self._name_scope(name, values=flat_values):
            return tf.reduce_prod(
                [dist.prob(val)
                 for dist, val in zip(self._flat_dists, flat_values)],
                axis=0)
