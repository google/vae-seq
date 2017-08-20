import numpy as np
import pickle
import tensorflow as tf
import sonnet as snt

from . import game as game_mod


class Environment(snt.RNNCore):
    def __init__(self, hparams, name=None):
        super(Environment, self).__init__(name or self.__class__.__name__)
        self._hparams = hparams

    @property
    def output_size(self):
        return tf.TensorShape(self._hparams.obs_shape)

    @property
    def output_dtype(self):
        return tf.int32

    @property
    def state_size(self):
        return tf.TensorShape([])  # pickled games.

    def initial_state(self, batch_size):
        hparams = self._hparams
        def _random_games(batch_size):
            return np.array(
                [pickle.dumps(game_mod.Game(hparams.test_game_width,
                                            hparams.test_game_classes))
                 for _ in range(batch_size)],
                dtype=object)
        state, = tf.py_func(_random_games, [batch_size], [tf.string])
        state.set_shape([None])
        return state

    def _build(self, actions, state):
        def _step(actions, state):
            games = [pickle.loads(game) for game in state]
            for action, game in zip(actions, games):
                game.take_action(action)
            obs = [game.render() for game in games]
            state = [pickle.dumps(game) for game in games]
            return np.array(obs, dtype=np.int32), np.array(state, dtype=object)
        out, state = tf.py_func(_step, [actions, state], [tf.int32, tf.string])
        out.set_shape([None] + self._hparams.obs_shape)
        state.set_shape([None])
        return out, state
