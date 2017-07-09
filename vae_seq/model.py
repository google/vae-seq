from . import vae as vae_mod

import abc
import re
import six
import tensorflow as tf

def in_model_graph(fn):
    """Decorates a method to execute in the context of the model's graph."""
    def ret(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)
    return ret


def with_default_session(make_session):
    """Decorates a method to create a session if it isn't passed in."""
    def wrapper(fn):
        def ret(self, *args, **kwargs):
            sess = kwargs.get('sess')
            with create_or_use(lambda: make_session(self), sess) as sess:
                kwargs['sess'] = sess
                return fn(self, *args, **kwargs)
        return ret
    return wrapper


class create_or_use(object):
    def __init__(self, make_session, sess=None):
        self._created = sess is None
        self._sess = make_session() if sess is None else sess
    def __enter__(self, *args, **kwargs):
        if self._created:
            return self._sess.__enter__(*args, **kwargs)
        return self._sess
    def __exit__(self, *args, **kwargs):
        if self._created:
            self._sess.__exit__(*args, **kwargs)


@six.add_metaclass(abc.ABCMeta)
class Model(object):
    def __init__(self, hparams, context, observed):
        vae = vae_mod.VAE_TYPES[hparams.vae_type](
            hparams,
            self.make_obs_encoder(hparams),
            self.make_obs_decoder(hparams))
        vae_tensors = vae(context, observed)
        divergence = tf.reduce_mean(vae_tensors.inf_kl, axis=0)
        tf.summary.scalar('divergence', divergence)
        inf_log_prob = tf.reduce_mean(vae_tensors.inf_log_prob, axis=0)
        tf.summary.scalar('inf_log_prob', inf_log_prob)
        elbo = inf_log_prob - divergence
        tf.summary.scalar('elbo', elbo)
        divergence_strength = tf.sigmoid(
            tf.to_float(tf.train.get_or_create_global_step()) /
            hparams.divergence_strength_halfway_point - 1.)
        tf.summary.scalar('divergence_strength', divergence_strength)
        elbo_opt = inf_log_prob - divergence * divergence_strength
        opt = tf.train.AdamOptimizer(hparams.learning_rate)
        grads_and_vars = opt.compute_gradients(-elbo_opt)
        for grad, var in grads_and_vars:
            tag = var.name.replace(':0', '')
            if grad is None:
                print 'WARNING: Gradient for ' + tag + ' is missing!'
                continue
            tf.summary.histogram(tag, var)
            tf.summary.histogram(tag + '/gradient', grad)
        train_op = opt.apply_gradients(
            grads_and_vars, global_step=tf.train.get_or_create_global_step())

        if hparams.check_numerics:
            tf.add_check_numerics_ops()

        graph = tf.get_default_graph()
        self.__dict__.update(locals())

    @in_model_graph
    def make_training_session(self):
        hparams = self.hparams
        logging_hook = tf.train.LoggingTensorHook(
            dict(step=tf.train.get_or_create_global_step(),
                 divergence=self.divergence,
                 inf_log_prob=self.inf_log_prob,
                 elbo=self.elbo,),
            every_n_secs=60. * 5)
        return tf.train.MonitoredTrainingSession(
            checkpoint_dir=hparams.logdir,
            is_chief=True,
            hooks=[logging_hook])

    @in_model_graph
    def make_eval_session(self):
        hparams = self.hparams
        session_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=hparams.logdir)
        return tf.train.MonitoredSession(session_creator=session_creator)

    @in_model_graph
    @with_default_session(make_training_session)
    def train(self, steps, sess=None):
        hparams = self.hparams
        for local_step in xrange(steps):
            if sess.should_stop():
                break
            sess.run(self.train_op)
            if local_step % 1000 == 0:
                self.sample_debug(sess=sess)

    @in_model_graph
    @with_default_session(make_eval_session)
    def grep_variables(self, regex, sess=None):
        hparams = self.hparams
        evals = {}
        for var in self.graph.get_collection('variables'):
            if re.search(regex, var.name) and 'Adam' not in var.name:
                evals[var.name] = var
        if not evals:
            print 'No variables matched.'
            return None
        return sess.run(evals)

    @in_model_graph
    @with_default_session(make_eval_session)
    def sample_debug(self, sess=None):
        hparams = self.hparams
        vals = sess.run({
            'step': tf.train.get_or_create_global_step(),
            'context': self.context,
            'gen_sample': self.vae_tensors.gen_x})
        print 'GLOBAL STEP:', vals['step']
        for i in range(min(5, hparams.batch_size)):
            self.display_sequence(vals['context'][i], vals['gen_sample'][i])

    @abc.abstractmethod
    def display_sequence(self, context, sample):
        raise NotImplementedError()

    @abc.abstractmethod
    def make_obs_encoder(self, hparams):
        raise NotImplementedError()

    @abc.abstractmethod
    def make_obs_decoder(self, hparams):
        raise NotImplementedError()
