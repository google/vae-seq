"""Functions to build up training and generation graphs."""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import sonnet as snt

from vaeseq import context as context_mod
from vaeseq import model as model_mod
from vaeseq import train as train_mod
from vaeseq import util
from vaeseq import vae as vae_mod

from . import agent as agent_mod
from . import codec as codec_mod
from . import environment


class Model(model_mod.ModelBase):
    """Putting everything together."""

    def __init__(self, hparams, session_params):
        self._env = environment.Environment(hparams)
        super(Model, self).__init__(hparams, session_params)

    def _make_encoder(self):
        return codec_mod.ObsEncoder(self.hparams)

    def _make_decoder(self):
        return codec_mod.ObsDecoder(self.hparams)

    def _make_inputs(self):
        return agent_mod.TrainableAgent(self.hparams, self.encoder)

    def _make_context(self):
        # Inputs are agent action logits; pass them through as context.
        input_encoder = codec_mod.InputEncoder(self.hparams)
        return context_mod.EncodeObserved(self.encoder,
                                          input_encoder=input_encoder)

    def _make_dataset(self, dataset):
        del dataset  #  Not used.
        cell = self._env
        cell = util.input_recording_rnn(
            cell,
            input_size=self.inputs.output_size)
        cell_output_dtype = (self.decoder.event_dtype,
                             self.inputs.output_dtype)
        cell_output_observations = lambda out: out[0]
        sequence_size = util.sequence_size(self.hparams)
        def _drive_env(agent, batch_size):
            cell_initial_state = self._env.initial_state(batch_size)
            observed, inputs = agent.drive_rnn(
                cell=cell,
                sequence_size=sequence_size,
                initial_state=agent.initial_state(batch_size),
                cell_initial_state=cell_initial_state,
                cell_output_dtype=cell_output_dtype,
                cell_output_observations=cell_output_observations)
            return inputs, observed

        batch_size = util.batch_size(self.hparams)
        train_batch_size = batch_size // 2
        random_batch_size = batch_size - train_batch_size
        inputs1, observed1 = _drive_env(self.inputs, train_batch_size)
        inputs2, observed2 = _drive_env(agent_mod.RandomAgent(self.hparams),
                                        random_batch_size)
        tf.summary.histogram("actions", tf.argmax(inputs1, axis=-1))
        inputs, observed = snt.nest.map(
            lambda t1, t2: tf.concat([t1, t2], axis=0),
            (inputs1, observed1),
            (inputs2, observed2))
        return inputs, observed

    def _make_output_summary(self, tag, observed):
        return tf.summary.scalar(
            tag + "/score",
            tf.reduce_mean(tf.reduce_sum(observed["score"], axis=1), axis=0),
            collections=[])

    def _make_elbo_trainer(self):
        global_step = tf.train.get_or_create_global_step()
        loss = train_mod.ELBOLoss(self.hparams, self.vae)
        def _variables():
            agent_vars = set(self.inputs.get_variables())
            return [var for var in tf.trainable_variables()
                    if var not in agent_vars]
        return train_mod.Trainer(self.hparams, global_step=global_step,
                                 loss=loss, variables=_variables,
                                 name="elbo_trainer")

    def _make_agent_trainer(self):
        global_step = None  # Do not increment global step twice per turn.
        loss = train_mod.RewardLoss(
            self.hparams, self.inputs, self.context, self.vae,
            reward=lambda observed: observed["score"])
        return train_mod.Trainer(self.hparams, global_step=global_step,
                                 loss=loss, variables=self.inputs.get_variables,
                                 name="agent_trainer")

    def _make_trainer(self):
        return train_mod.Group([self._make_elbo_trainer(),
                                self._make_agent_trainer()])
