"""Functions to build up training and generation graphs."""

import tensorflow as tf
import sonnet as snt

from vae_seq import hparams as hparams_mod
from vae_seq import train as train_mod
from vae_seq import util
from vae_seq import vae as vae_mod

from . import agent as agent_mod
from . import codec


def make_vae(hparams):
    """Constructs a VAE for modeling audio."""
    with tf.name_scope("vae"):
        obs_encoder = codec.AudioObsEncoder(hparams)
        obs_decoder = codec.AudioObsDecoder(hparams)
        agent = agent_mod.Agent(hparams, obs_encoder)
        return vae_mod.make(hparams, agent, obs_encoder, obs_decoder)


def train_graph(hparams, vae, observed, offsets):
    """Constructs a training graph."""
    with tf.name_scope("training_data"):
        agent_inputs = agent_mod.timing_input(hparams, offsets)
        contexts = agent_mod.contexts_for_static_observations(
            observed,
            vae.agent,
            agent_inputs)
    train_ops = train_mod.TrainOps(hparams, vae)
    train_op, debug_tensors = train_ops(contexts, observed)
    return train_op, debug_tensors


def gen_graph(hparams, vae):
    """Samples audio from the trained VAE."""
    with tf.name_scope("generate"):
        agent_inputs = agent_mod.timing_input(hparams)
        return vae.gen_core.generate(agent_inputs)[0]


def flatten_samples(hparams, sequences):
    """Flatten a batch of observation sequences into a batch of samples."""
    return tf.reshape(
        sequences,
        [util.batch_size(hparams),
         util.sequence_size(hparams) * hparams.samples_per_step])


def train(hparams, dataset, log_dir, num_steps):
    """Trains/continues training a VAE and saves it in log_dir."""
    iterator = tf.contrib.data.Iterator.from_dataset(dataset)
    observed, offsets = iterator.get_next()
    vae = make_vae(hparams)
    train_op, debug_tensors = train_graph(hparams, vae, observed, offsets)
    generated = gen_graph(hparams, vae)
    debug_tensors["step"] = tf.train.get_or_create_global_step()
    logging_hook = tf.train.LoggingTensorHook(debug_tensors, every_n_secs=60.)
    train_audio = tf.summary.audio(
        "observed",
        flatten_samples(hparams, observed),
        hparams.audio_rate,
        collections=[])
    gen_audio = tf.summary.audio(
        "generated",
        flatten_samples(hparams, generated),
        hparams.audio_rate,
        collections=[])
    display_hook = tf.train.SummarySaverHook(
        save_steps=1000, output_dir=log_dir,
        summary_op=tf.summary.merge([train_audio, gen_audio]))
    local_init_op = tf.group(
        tf.local_variables_initializer(),
        tf.tables_initializer(),
        iterator.initializer)
    scaffold = tf.train.Scaffold(
        local_init_op=local_init_op)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=log_dir,
        is_chief=True,
        scaffold=scaffold,
        hooks=[logging_hook, display_hook]) as sess:
        for _ in xrange(num_steps):
            if sess.should_stop():
                break
            sess.run(train_op)
