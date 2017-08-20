import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import distributions

from . import base
from .. import util

class RNN(base.VAEBase):
    """We can view an RNN as a VAE with no latent variables:

    Notation:
     - d_1:T are the (deterministic) RNN outputs.
     - x_1:T are the observed states.
     - c_1:T are per-timestep contexts.

            Generative model
          =====================
          x_1               x_t
           ^                 ^
           |                 |
          d_1 ------------> d_t
           ^                 ^
           |                 |
          c_1               c_t
    """

    def _allocate(self):
        hparams = self._hparams
        self._d_core = util.make_rnn(hparams, name="d_core")
        self._z_distcore = NoLatents(hparams)
        self._x_distcore = ObsDist(hparams, self._d_core, self._obs_decoder)

    @property
    def latent_prior_distcore(self):
        return self._z_distcore

    @property
    def observed_distcore(self):
        return self._x_distcore

    def infer_latents(self, contexts, observed):
        batch_size, length = tf.unstack(tf.shape(observed)[:2])
        latents = tf.zeros([batch_size, length, 0])
        divs = tf.zeros([batch_size, length])
        latents.set_shape(observed.get_shape()[:2].concatenate([0]))
        divs.set_shape(observed.get_shape()[:2])
        return latents, divs


class ObsDist(base.DistCore):
    def __init__(self, hparams, d_core, obs_decoder, name=None):
        super(ObsDist, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams
        self._d_core = d_core
        self._obs_decoder = obs_decoder

    @property
    def state_size(self):
        return self._d_core.state_size

    @property
    def event_size(self):
        return tf.TensorShape(self._hparams.obs_shape)

    @property
    def event_dtype(self):
        return self._obs_decoder.event_dtype

    def _build_dist(self, (context, z), d_state):
        d, d_state = self._d_core(util.concat_features(context), d_state)
        return self._obs_decoder.dist(d), d_state

    def _next_state(self, d_state, event=None):
        return d_state


class NoEventsDist(distributions.Distribution):
    def __init__(self, batch_shape, dtype=tf.float32, name=None):
        super(NoEventsDist, self).__init__(
            dtype=dtype,
            reparameterization_type=distributions.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False,
            name=name or self.__class__.__name__)
        self._batch_shape_val = tf.TensorShape(batch_shape)

    def _batch_shape(self):
        return self._batch_shape_val

    def _sample_n(self, n, seed=None):
        shape = [n] + self.batch_shape.as_list() + [0]
        return tf.zeros(shape)


class NoLatents(base.DistCore):
    def __init__(self, hparams, name=None):
        super(NoLatents, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams

    @property
    def state_size(self):
        return ()

    @property
    def event_size(self):
        return tf.TensorShape([0])

    @property
    def event_dtype(self):
        return tf.float32

    def _build_dist(self, context, state):
        hparams = self._hparams
        dist = NoEventsDist([hparams.batch_size])
        return dist, ()

    def _next_state(self, state_arg, event=None):
        return ()
