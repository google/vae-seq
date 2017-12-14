"""The model for MIDI music.

At each time step, we predict a pair of:
* 128 independent Beta variables, corresponding to each note.
* K, a Binomial variable counting the number of notes played.

When generating music, we emit the top K notes per timestep.
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf

from vaeseq import codec as codec_mod
from vaeseq import model as model_mod
from vaeseq import util

from . import dataset as dataset_mod


class Model(model_mod.ModelBase):
    """Putting everything together."""

    def _make_encoder(self):
        """Constructs an encoder for a single observation."""
        return codec_mod.MLPObsEncoder(self.hparams, name="obs_encoder")

    def _make_decoder(self):
        """Constructs a decoder for a single observation."""
        # We need 2 * 128 (note beta) + 1 (count binomial) parameters.
        params = util.make_mlp(
            self.hparams,
            self.hparams.obs_decoder_fc_hidden_layers + [128 * 2 + 1])
        positive_projection = util.positive_projection(self.hparams)
        def _split_params(inp):
            conc1, conc0, logits = tf.split(inp, [128, 128, 1], axis=-1)
            conc1 = positive_projection(conc1) + 1e-4
            conc0 = positive_projection(conc0) + 1e-4
            return ((conc1, conc0), logits)
        single_note_decoder = codec_mod.BetaDecoder()
        notes_decoder = codec_mod.BatchDecoder(
            single_note_decoder, event_size=[128], name="notes_decoder")
        count_decoder = codec_mod.BinomialDecoder(total_count=128,
                                                  squeeze_input=True,
                                                  name="count_decoder")
        full_decoder = codec_mod.GroupDecoder((notes_decoder, count_decoder))
        return codec_mod.DecoderSequence(
            [params, _split_params], full_decoder, name="decoder")

    def _make_dataset(self, files):
        dataset = dataset_mod.piano_roll_sequences(
            files,
            util.batch_size(self.hparams),
            util.sequence_size(self.hparams),
            rate=self.hparams.rate)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, iterator.initializer)
        piano_roll = iterator.get_next()
        shape = tf.shape(piano_roll)
        notes = tf.where(piano_roll, tf.fill(shape, 0.95), tf.fill(shape, 0.05))
        counts = tf.reduce_sum(tf.to_int32(piano_roll), axis=-1)
        observed = (notes, counts)
        inputs = None
        return inputs, observed

    # Samples per second when generating audio output.
    SYNTHESIZED_RATE = 16000
    def _render(self, observed):
        """Returns a batch of wave forms corresponding to the observations."""
        notes, counts = observed

        def _synthesize(notes, counts):
            """Use pretty_midi to synthesize a wave form."""
            piano_roll = np.zeros((len(counts), 128), dtype=np.bool)
            top_notes = np.argsort(notes)
            for roll_t, top_notes_t, k in zip(piano_roll, top_notes, counts):
                for i in top_notes_t[-k:]:
                    roll_t[i] = True
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
        def _map_batch_elem(notes_counts):
            notes, counts = notes_counts
            return tf.py_func(_synthesize, [notes, counts], [tf.float32])[0]
        return tf.map_fn(_map_batch_elem, (notes, counts), dtype=tf.float32)

    def _make_output_summary(self, tag, observed):
        notes, counts = observed
        return tf.summary.merge(
            [tf.summary.audio(
                tag + "/audio",
                self._render(observed),
                self.SYNTHESIZED_RATE,
                collections=[]),
             tf.summary.scalar(
                 tag + "/note_avg",
                 tf.reduce_mean(notes)),
             tf.summary.scalar(
                 tag + "/note_count",
                 tf.reduce_mean(tf.to_float(counts)))])
