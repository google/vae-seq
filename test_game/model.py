import threading
import numpy as np
import tensorflow as tf
from . import game as game_mod
from .. import obs_layers
from .. import model


class Model(model.Model):
    def __init__(self, hparams):
        hparams.obs_shape = [hparams.test_game_classes]
        game = game_mod.Game(hparams.test_game_width, hparams.test_game_classes)
        batches = game_mod.produce_batches(
            game, hparams.batch_size, hparams.sequence_size)
        lock = threading.Lock()
        def _next_batch():
            with lock:
                return batches.next()
        actions, observed = tf.py_func(
            _next_batch,
            [],
            [tf.int32, tf.float32],
            name='produce_batches')
        actions.set_shape([hparams.batch_size, hparams.sequence_size])
        observed.set_shape([hparams.batch_size, hparams.sequence_size,
                            hparams.test_game_classes])
        one_hot_actions = tf.one_hot(
            actions - 1,
            len(game_mod.ACTIONS) - 1,
            dtype=tf.float32)
        super(Model, self).__init__(
            hparams,
            context=one_hot_actions,
            observed=observed)

    def display_sequence(self, context, sample):
        def _context_step_to_action(context_t):
            indices, = np.nonzero(context_t)
            assert indices.size in (0, 1), 'Context step is not one-hot.'
            if indices.size == 0:
                return 0
            return 1 + indices[0]
        actions = [_context_step_to_action(ct) for ct in context]
        game_mod.print_sequence(actions, sample)

    def make_obs_encoder(self, hparams):
        return obs_layers.ObsEncoder(hparams)

    def make_obs_decoder(self, hparams):
        return obs_layers.ObsDecoder(hparams)
