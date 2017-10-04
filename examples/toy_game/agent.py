"""An agent implementation for the test game environment."""

import numpy as np
import tensorflow as tf
from vae_seq import agent as agent_mod
from vae_seq import util

from . import game as game_mod


class Agent(agent_mod.Agent):
    """Outputs the context based on the previous observation.

    This implementation uses a one-hot representation of the next
    action (chosen randomly by default, or read from stdin in
    interactive mode) as well as an encoding of the previous
    observation.
    """

    def __init__(self, hparams, obs_encoder, name=None):
        super(Agent, self).__init__(name=name)
        self.interactive = False
        self._hparams = hparams
        with self._enter_variable_scope():
            self._inner_agent = agent_mod.EncodeObsAgent(obs_encoder)

    @property
    def context_size(self):
        return (tf.TensorShape([len(game_mod.ACTIONS) - 1]),
                self._inner_agent.context_size)

    @property
    def context_dtype(self):
        return (tf.float32, self._inner_agent.context_dtype)

    @property
    def state_size(self):
        return (
            tf.TensorShape([]),  # selected action.
            self._inner_agent.state_size,)  # inner agent state.

    @property
    def state_dtype(self):
        return (tf.int32, self._inner_agent.state_dtype)

    def initial_state(self, batch_size):
        return (
            tf.zeros([batch_size], dtype=tf.int32),  # NOOPs.
            self._inner_agent.initial_state(batch_size),)

    def observe(self, agent_input, observation, state):
        batch_size = util.batch_size(self._hparams)
        _unused_prev_action, inner_state = state
        if not self.interactive:
            actions = tf.random_uniform(
                [batch_size], 0, len(game_mod.ACTIONS), dtype=tf.int32)
        else:
            single_action, = tf.py_func(
                input_action, [observation[0, :]], [tf.int32])
            single_action.set_shape([])
            actions = tf.tile(tf.expand_dims(single_action, 0), [batch_size])
        inner_state = self._inner_agent.observe(agent_input, observation,
                                                inner_state)
        return (actions, inner_state)

    def context(self, agent_input, state):
        actions, inner_state = state
        one_hot_action = tf.one_hot(
            actions - 1, len(game_mod.ACTIONS) - 1, dtype=tf.float32)
        return (one_hot_action,
                self._inner_agent.context(agent_input, inner_state))

    def env_input(self, agent_input, state):
        """Input for the actual game, used to generate training data."""
        actions, _unused_inner_state = state
        return actions

    def inputs(self):
        return agent_mod.null_inputs(
            util.batch_size(self._hparams),
            util.sequence_size(self._hparams))

    def contexts_and_observations(self, env):
        return agent_mod.contexts_and_observations_from_environment(
            env, self, self.inputs())


def input_action(obs):
    print("OBSERVATION:", list(obs))
    action_menu = ("[" + ", ".join(
        [str(i) + ": " + action
         for (i, action) in enumerate(game_mod.ACTIONS)]) + "]")
    while True:
        try:
            action = np.int32(
                raw_input("Pick an action: " + action_menu + "> "))
            assert action >= 0 and action < len(game_mod.ACTIONS)
            break
        except (AssertionError, ValueError):
            print("Please enter a valid action.")
    return action
