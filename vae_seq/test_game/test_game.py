import tensorflow as tf
from vae_seq import hparams as hparams_mod
from vae_seq.test_game import model as model_mod

flags = tf.app.flags
flags.DEFINE_string('hparams', '', 'HParams overrides.')
flags.DEFINE_integer('iters', 5000, 'Number of training iterations')

FLAGS = flags.FLAGS

hparams = hparams_mod.HParams()
hparams.parse(FLAGS.hparams)

model = model_mod.Model(hparams)
model.train(FLAGS.iters)
