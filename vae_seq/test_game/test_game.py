import tensorflow as tf
from vae_seq import hparams as hparams_mod
from vae_seq.test_game import model as model_mod

flags = tf.app.flags
flags.DEFINE_string('hparams', '', 'HParams overrides.')

hparams = hparams_mod.HParams(**eval('dict(' + flags.FLAGS.hparams + ')'))
model = model_mod.Model(hparams)
model.train(5000)
