"""Functions to build up training and generation graphs."""

from __future__ import print_function

import numpy as np
import tensorflow as tf

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

    def _make_vae(self):
        obs_encoder = codec_mod.ObsEncoder(self.hparams)
        obs_decoder = codec_mod.ObsDecoder(self.hparams)
        agent = agent_mod.TrainableAgent(self.hparams, obs_encoder)
        return vae_mod.make(self.hparams, agent, obs_encoder, obs_decoder)

    def _open_dataset(self, dataset):
        del dataset  #  Not used.
        agent = self.vae.agent
        return agent.contexts_and_observations_from_environment(
            self._env,
            agent.get_inputs(util.batch_size(self.hparams),
                             util.sequence_size(self.hparams)))

    def _make_agent_loss(self):
        if self.hparams.train_agent:
            return train_mod.AgentLoss(self.hparams, self.vae)
        return None

    def _make_output_summary(self, tag, observed):
        return tf.summary.scalar(
            tag + "/score",
            tf.reduce_sum(observed["score"]),
            collections=[])
