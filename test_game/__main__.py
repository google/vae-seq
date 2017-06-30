import tensorflow as tf
from . import model as model_mod
from .. import hparams as hparams_mod

flags = tf.app.flags
flags.DEFINE_string('hparams', '', 'HParams overrides.')

hparams = hparams_mod.HParams(**eval('dict(' + flags.FLAGS.hparams + ')'))
model = model_mod.Model(hparams)
model.train(5000)
