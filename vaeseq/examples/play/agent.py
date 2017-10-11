"""Game-playing agent."""

import sonnet as snt
import tensorflow as tf

from vaeseq import agent as agent_mod
from vaeseq import util


class TrainableAgent(agent_mod.Agent):
    """An agent where the context is both an encoding of the previous
       observation and the parameters of an action policy.
    """

    def __init__(self, hparams, obs_encoder, name=None):
        super(TrainableAgent, self).__init__(name=name)
        self._hparams = hparams
        self._obs_encoder = obs_encoder
        self._action_size = tf.TensorShape([self._hparams.game_action_space])
        self._agent_variables = None
        with self._enter_variable_scope():
            self._policy_rnn = util.make_rnn(hparams, name="policy_rnn")
            self._project_act = util.make_mlp(
                hparams, layers=[hparams.game_action_space], name="policy_proj")

    @property
    def context_size(self):
        return (self._action_size,
                self._obs_encoder.output_size)

    @property
    def context_dtype(self):
        return (tf.float32, tf.float32)

    @property
    def state_size(self):
        return (self._policy_rnn.state_size,
                self._action_size,
                self._obs_encoder.output_size)

    @property
    def state_dtype(self):
        return (tf.float32, tf.float32, tf.float32)

    def initial_state(self, batch_size):
        return snt.nest.map(
            lambda size: tf.zeros([batch_size] +
                                  tf.TensorShape(size).as_list()),
            self.state_size)

    def agent_variables(self):
        if self._agent_variables is None:
            raise ValueError("Agent variables haven't been constructed yet.")
        return self._agent_variables

    def rewards(self, observed):
        if self._hparams.train_agent:
            return observed["score"]
        return None

    def observe(self, agent_input, observation, state):
        del agent_input  # Unused.
        obs_enc = self._obs_encoder(observation)
        rnn_state = state[0]
        hidden, rnn_state = self._policy_rnn(obs_enc, rnn_state)
        action_logits = self._project_act(hidden)
        self._agent_variables = (self._policy_rnn.get_variables(),
                                 self._project_act.get_variables())
        return (rnn_state, action_logits, obs_enc)

    def context(self, agent_input, state):
        del agent_input  # Unused.
        return (state[1], state[2])

    def env_input(self, agent_input, state):
        return tf.distributions.Categorical(logits=state[1]).sample()
