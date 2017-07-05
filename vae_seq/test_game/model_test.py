import tensorflow as tf

from vae_seq import hparams as hparams_mod
from vae_seq.test_game import model as model_mod


class ModelTest(tf.test.TestCase):

    def monitored_test_session(self, *args, **kwargs):
        return tf.train.MonitoredSession(
            session_creator=TestSessionCreator(
                lambda: self.test_session(*args, **kwargs)))

    def _build_model(self, **params):
        hparams = hparams_mod.HParams(
            logdir=None,
            batch_size=100,
            sequence_size=5,
            **params)
        return model_mod.Model(hparams)

    def testConstruction(self):
        self._build_model()

    def testTrainingSRNN(self):
        model = self._build_model(vae_type='SRNN')
        sess = model.make_training_session()
        elbo_1 = sess.run(model.elbo)
        model.train(1000, sess=sess)
        elbo_2 = sess.run(model.elbo)
        self.assertGreater(elbo_2, elbo_1)

    def testTrainingIndependentSequence(self):
        model = self._build_model(vae_type='IndependentSequence')
        sess = model.make_training_session()
        elbo_1 = sess.run(model.elbo)
        model.train(1000, sess=sess)
        elbo_2 = sess.run(model.elbo)
        self.assertGreater(elbo_2, elbo_1)


if __name__ == '__main__':
    tf.test.main()
