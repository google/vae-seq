"""Functions to build up training and generation graphs."""

from __future__ import print_function

import sonnet as snt
import tensorflow as tf

from vaeseq import agent as agent_mod
from vaeseq import codec as codec_mod
from vaeseq import model as model_mod
from vaeseq import util
from vaeseq import vae as vae_mod

from . import dataset as dataset_mod


class Model(model_mod.ModelBase):
    """Putting everything together."""

    def __init__(self, hparams, session_params, vocab_corpus):
        self._char_to_id, self._id_to_char = dataset_mod.vocabulary(
            vocab_corpus,
            max_size=hparams.vocab_size,
            num_oov_buckets=hparams.oov_buckets)
        super(Model, self).__init__(hparams, session_params)

    def _make_obs_encoder(self):
        """Constructs an encoding for a single character ID."""
        embed = snt.Embed(
            vocab_size=self.hparams.vocab_size + self.hparams.oov_buckets,
            embed_dim=self.hparams.embed_size)
        mlp = codec_mod.MLPObsEncoder(self.hparams)
        return codec_mod.EncoderSequence([embed, mlp], name="obs_encoder")

    def _make_obs_decoder(self):
        """Constructs a decoding for a single character ID."""
        return codec_mod.MLPObsDecoder(
            self.hparams,
            decoder=codec_mod.CategoricalDecoder(),
            param_size=self.hparams.vocab_size + self.hparams.oov_buckets,
            name="obs_decoder")

    def _make_vae(self):
        obs_encoder = self._make_obs_encoder()
        obs_decoder = self._make_obs_decoder()
        agent = agent_mod.EncodeObsAgent(obs_encoder)
        return vae_mod.make(self.hparams, agent, obs_encoder, obs_decoder)

    def _open_dataset(self, corpus):
        dataset = dataset_mod.characters(corpus,
                                         util.batch_size(self.hparams),
                                         util.sequence_size(self.hparams))
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, iterator.initializer)
        observed = self._char_to_id.lookup(iterator.get_next())
        contexts = self.vae.agent.contexts_for_static_observations(observed)
        return contexts, observed

    def _make_output_summary(self, tag, observed):
        return tf.summary.text(tag, self._render(observed), collections=[])

    def _render(self, observed):
        """Returns a batch of strings corresponding to the ID sequences."""
        # Note, tf.reduce_sum doesn't work on strings.
        return tf.py_func(lambda chars: chars.sum(axis=-1),
                          [self._id_to_char.lookup(tf.to_int64(observed))],
                          [tf.string])[0]

    def generate(self):
        """Return UTF-8 strings rather than bytes."""
        for string in super(Model, self).generate():
            yield tf.compat.as_text(string)
