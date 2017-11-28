"""Tests for environment.py."""

import numpy as np
import tensorflow as tf

from vaeseq import util
from vaeseq.examples.play import environment as env_mod


class EnvironmentTest(tf.test.TestCase):

    def test_environment(self):
        hparams = tf.contrib.training.HParams(
            game="CartPole-v0",
            game_output_size=[4])
        left_logits = [-100., 100.]
        right_logits = [100., -100.]
        actions = [[left_logits, right_logits] * 10,
                   [right_logits, left_logits] * 10]
        batch_size = len(actions)
        actions = tf.constant(actions, dtype=tf.float32)
        env = env_mod.Environment(hparams)
        initial_state = env.initial_state(batch_size=batch_size)
        output_dtypes = env.output_dtype
        observed, _ = util.heterogeneous_dynamic_rnn(
            env, actions,
            initial_state=initial_state,
            output_dtypes=output_dtypes)
        with self.test_session() as sess:
            observed = sess.run(observed)
            outputs = observed["output"]
            game_over = observed["game_over"]
            scores = observed["score"]
            self.assertTrue(np.all(scores[game_over > 0] == 1))
            nonzero_gameover_scores = scores[np.nonzero(scores[game_over > 0])]
            nonzero_gameover_outs = outputs[np.nonzero(outputs[game_over > 0])]
            self.assertLessEqual(len(nonzero_gameover_scores), batch_size)
            self.assertLessEqual(len(nonzero_gameover_outs), batch_size * 4)


if __name__ == "__main__":
    tf.test.main()
