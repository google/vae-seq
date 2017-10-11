"""Environment that runs OpenAI Gym games."""

import threading

import gym
import numpy as np
import sonnet as snt
import tensorflow as tf

from vaeseq import agent as agent_mod
from vaeseq import util


class Environment(agent_mod.Environment):
    """Plays a batch of games."""

    def __init__(self, hparams, name=None):
        super(Environment, self).__init__(name=name)
        self._hparams = hparams
        self._games = {}
        self._next_id = 1
        self._id_lock = threading.Lock()

    @property
    def output_size(self):
        return dict(output=tf.TensorShape(self._hparams.game_output_size),
                    score=tf.TensorShape([]),
                    game_over=tf.TensorShape([]),)

    @property
    def output_dtype(self):
        return dict(output=tf.float32,
                    score=tf.float32,
                    game_over=tf.bool)

    @property
    def state_size(self):
        """The state is a game ID, or 0 if the game is over."""
        return tf.TensorShape([])

    @property
    def state_dtype(self):
        """The state is a game ID, or -1 if the game is over."""
        return tf.int64

    def initial_state(self, batch_size):
        def _make_games(batch_size):
            """Produces a serialized batch of randomized games."""
            with self._id_lock:
                first_id = self._next_id
                self._next_id += batch_size
                game_ids = range(first_id, self._next_id)
            for game_id in game_ids:
                game = gym.make(self._hparams.game)
                game.reset()
                self._games[game_id] = game
            return np.asarray(game_ids, dtype=np.int64)

        state, = tf.py_func(_make_games, [batch_size], [tf.int64])
        state.set_shape([None])
        return state

    def _build(self, actions, state):
        def _step_games(actions, state):
            """Take a step in a single game."""
            score = np.zeros(len(state), dtype=np.float32)
            output = np.zeros([len(state)] + self._hparams.game_output_size,
                           dtype=np.float32)
            for i in np.nonzero(state)[0]:
                game = self._games[state[i]]
                output[i], score[i], game_over, _ = game.step(actions[i])
                if game_over:
                    del self._games[state[i]]
                    state[i] = 0
            game_over = state == 0
            return output, score, game_over, state

        output, score, game_over, state = tf.py_func(
            _step_games, [actions, state],
            [tf.float32, tf.float32, tf.bool, tf.int64])
        output = dict(output=output,
                      score=score,
                      game_over=game_over)
        # Fix up the inferred shapes.
        util.set_tensor_shapes(output, self.output_size, add_batch_dim=True)
        util.set_tensor_shapes(state, self.state_size, add_batch_dim=True)
        return output, state
