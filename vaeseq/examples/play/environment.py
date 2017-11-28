"""Environment that runs OpenAI Gym games."""

import threading
import time

import gym
import numpy as np
import sonnet as snt
import tensorflow as tf

from vaeseq import util


class Environment(snt.RNNCore):
    """Plays a batch of games."""

    def __init__(self, hparams, name=None):
        super(Environment, self).__init__(name=name)
        self._hparams = hparams
        self._games = {}
        self._games_lock = threading.Lock()
        self._next_id = 1
        self._id_lock = threading.Lock()
        self._step_time = None
        self._render_thread = None

    @property
    def output_size(self):
        return dict(output=tf.TensorShape(self._hparams.game_output_size),
                    score=tf.TensorShape([]),
                    game_over=tf.TensorShape([]))

    @property
    def output_dtype(self):
        return dict(output=tf.float32,
                    score=tf.float32,
                    game_over=tf.float32)

    @property
    def state_size(self):
        """The state is a game ID, or 0 if the game is over."""
        return tf.TensorShape([])

    @property
    def state_dtype(self):
        """The state is a game ID, or 0 if the game is over."""
        return tf.int64

    def initial_state(self, batch_size):
        def _make_games(batch_size):
            """Produces a serialized batch of randomized games."""
            with self._id_lock:
                first_id = self._next_id
                self._next_id += batch_size
                game_ids = range(first_id, self._next_id)
            updates = []
            for game_id in game_ids:
                game = gym.make(self._hparams.game)
                game.reset()
                updates.append((game_id, game))
            with self._games_lock:
                self._games.update(updates)
            return np.asarray(game_ids, dtype=np.int64)

        state, = tf.py_func(_make_games, [batch_size], [tf.int64])
        state.set_shape([None])
        return state

    def _build(self, input_, state):
        actions = tf.distributions.Categorical(logits=input_).sample()

        def _step_games(actions, state):
            """Take a step in a single game."""
            score = np.zeros(len(state), dtype=np.float32)
            output = np.zeros([len(state)] + self._hparams.game_output_size,
                              dtype=np.float32)
            games = [None] * len(state)
            with self._games_lock:
                for i, game_id in enumerate(state):
                    if game_id:
                        games[i] = self._games[game_id]
            finished_games = []
            for i, game in enumerate(games):
                if game is None:
                    continue
                output[i], score[i], game_over, _ = game.step(actions[i])
                if game_over:
                    finished_games.append(state[i])
                    state[i] = 0
            if finished_games:
                with self._games_lock:
                    for game_id in finished_games:
                        del self._games[game_id]
            if self._render_thread is not None:
                time.sleep(0.1)
            return output, score, state

        output, score, state = tf.py_func(
            _step_games, [actions, state],
            [tf.float32, tf.float32, tf.int64])
        output = dict(output=output, score=score,
                      game_over=2. * tf.to_float(tf.equal(state, 0)) - 1.)
        # Fix up the inferred shapes.
        util.set_tensor_shapes(output, self.output_size, add_batch_dims=1)
        util.set_tensor_shapes(state, self.state_size, add_batch_dims=1)
        return output, state

    def start_render_thread(self):
        if self._render_thread is not None:
            return self._render_thread
        self._render_thread = threading.Thread(target=self._render_games_loop)
        self._render_thread.start()

    def stop_render_thread(self):
        if self._render_thread is None:
            return
        tmp = self._render_thread
        self._render_thread = None
        tmp.join()

    def _render_games_loop(self):
        while (self._render_thread is not None and
               threading.current_thread().ident == self._render_thread.ident):
            with self._games_lock:
                games = list(self._games.values())
            for game in games:
                game.render()
            time.sleep(0.05)
