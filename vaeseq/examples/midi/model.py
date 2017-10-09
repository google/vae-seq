"""Functions to build up training and generation graphs."""

from __future__ import print_function

import numpy as np
import tensorflow as tf

from vaeseq import agent as agent_mod
from vaeseq import codec as codec_mod
from vaeseq import model as model_mod
from vaeseq import util
from vaeseq import vae as vae_mod

from . import dataset as dataset_mod


class Model(model_mod.ModelBase):
    """Putting everything together."""

    def _make_obs_encoder(self):
        """Constructs an encoder for a single observation."""
        return codec_mod.MLPObsEncoder(self.hparams, name="obs_encoder")

    def _make_obs_decoder(self):
        """Constructs a decoder for a single observation."""
        note_decoder = codec_mod.BernoulliDecoder(dtype=tf.bool)
        return codec_mod.BatchDecoder(
            codec_mod.MLPObsDecoder(self.hparams, note_decoder, 128),
            event_size=[128], name="obs_decoder")

    def _make_vae(self):
        """Constructs a VAE for modeling character sequences."""
        obs_encoder = self._make_obs_encoder()
        obs_decoder = self._make_obs_decoder()
        agent = agent_mod.EncodeObsAgent(obs_encoder)
        return vae_mod.make(self.hparams, agent, obs_encoder, obs_decoder)

    def _open_dataset(self, files):
        dataset = dataset_mod.piano_roll_sequences(
            files,
            util.batch_size(self.hparams),
            util.sequence_size(self.hparams),
            rate=self.hparams.rate)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, iterator.initializer)
        observed = iterator.get_next()
        batch_size, sequence_size = tf.unstack(tf.shape(observed)[:2])
        contexts = agent_mod.contexts_for_static_observations(
            observed,
            self.vae.agent,
            agent_inputs=self._agent_inputs(batch_size, sequence_size))
        return contexts, observed

    # Samples per second when generating audio output.
    SYNTHESIZED_RATE = 16000
    def _render(self, observed):
        """Returns a batch of wave forms corresponding to the observations."""

        def _synthesize_roll(piano_roll):
            """Use pretty_midi to synthesize a wave form."""
            rate = self.hparams.rate
            midi = dataset_mod.piano_roll_to_midi(piano_roll, rate)
            wave = midi.synthesize(self.SYNTHESIZED_RATE)
            wave_len = len(wave)
            expect_len = (len(piano_roll) * self.SYNTHESIZED_RATE) // rate
            if wave_len < expect_len:
                wave = np.pad(wave, [0, expect_len - wave_len], mode='constant')
            else:
                wave = wave[:expect_len]
            return np.float32(wave)

        # Apply synthesize_roll on all elements of the batch.
        return tf.map_fn(
            lambda roll: tf.py_func(_synthesize_roll, [roll], [tf.float32])[0],
            observed, dtype=tf.float32)

    def _make_output_summary(self, tag, observed):
        return tf.summary.audio(
            tag,
            self._render(observed),
            self.SYNTHESIZED_RATE,
            collections=[])

    def _agent_inputs(self, batch_size, sequence_size):
        return agent_mod.null_inputs(batch_size, sequence_size)
