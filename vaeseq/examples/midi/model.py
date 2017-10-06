"""Functions to build up training and generation graphs."""

from __future__ import print_function
from builtins import range

import os.path
import pretty_midi
import numpy as np
import scipy.io.wavfile
import tensorflow as tf

from vaeseq import agent as agent_mod
from vaeseq import train as train_mod
from vaeseq import util
from vaeseq import vae as vae_mod

from . import codec

# Samples per second when generating audio output.
SYNTHESIZED_RATE = 16000


def make_vae(hparams, name=None):
    """Constructs a VAE for modeling character sequences."""
    with tf.name_scope(name, "vae"):
        obs_encoder = codec.ObsEncoder(hparams)
        obs_decoder = codec.ObsDecoder(hparams)
        agent = agent_mod.EncodeObsAgent(obs_encoder)
        return vae_mod.make(hparams, agent, obs_encoder, obs_decoder)


def train_graph(hparams, vae, observed, name=None):
    """Constructs a training graph."""
    with tf.name_scope(name, "training_data"):
        agent_inputs = agent_mod.null_inputs(
            util.batch_size(hparams),
            util.sequence_size(hparams))
        context = agent_mod.contexts_for_static_observations(
            observed,
            vae.agent,
            agent_inputs)
    train_ops = train_mod.TrainOps(hparams, vae)
    train_op, debug_tensors = train_ops(context, observed)
    return train_op, debug_tensors


def gen_graph(hparams, vae, name=None):
    """Samples strings from the trained VAE."""
    with tf.name_scope(name, "generate"):
        agent_inputs = agent_mod.null_inputs(
            util.batch_size(hparams),
            util.sequence_size(hparams))
        return vae.gen_core.generate(agent_inputs)[0]


def eval_graph(hparams, vae, observed, name=None):
    """Return mean log-prob(observed) and adds the update op to a collection."""
    with tf.name_scope(name, "evaluate"):
        agent_inputs = agent_mod.null_inputs(
            util.batch_size(hparams),
            util.sequence_size(hparams))
        log_probs = vae.eval_core.log_probs(agent_inputs, observed)
        mean, update_mean = tf.metrics.mean(log_probs, name="mean_log_prob")
        tf.summary.scalar("mean_log_prob", mean)
        tf.add_to_collection("metric_updates", update_mean)
        return mean


def synthesize(hparams, observed):
    """Returns a batch of wave forms corresponding to the observations."""

    def _synthesize_roll(piano_roll):
        """Use pretty_midi to synthesize a wave form."""
        # Convert the piano roll to a PrettyMIDI object, similar to
        # http://github.com/craffel/examples/reverse_pianoroll.py
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(0)
        padded_roll = np.pad(piano_roll, [(1, 1), (0, 0)], mode='constant')
        changes = np.diff(padded_roll, axis=0)
        notes = np.full(piano_roll.shape[1], -1, dtype=np.int)
        rate = float(hparams.rate)
        for tick, pitch in zip(*np.where(changes)):
            prev = notes[pitch]
            if prev == -1:
                notes[pitch] = tick
                continue
            notes[pitch] = -1
            instrument.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=prev / rate,
                end=tick / rate))
        midi.instruments.append(instrument)
        expected_length = int(len(piano_roll) / rate * SYNTHESIZED_RATE)
        wave = np.float32(midi.synthesize(SYNTHESIZED_RATE))
        length = len(wave)
        if length < expected_length:
            return np.pad(wave, [0, expected_length - length], mode='constant')
        elif length > expected_length:
            return wave[:expected_length]
        return wave

    # Apply synthesize_roll on all elements of the batch.
    return tf.map_fn(
        lambda roll: tf.py_func(_synthesize_roll, [roll], [tf.float32])[0],
        observed, dtype=tf.float32)


def observations(dataset):
    """Returns a batch of observations (piano rolls) from the dataset."""
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, iterator.initializer)
    return iterator.get_next()


def make_scaffold():
    """Session scaffold with all of the local init ops."""
    local_init_op = tf.group(
        tf.local_variables_initializer(),
        tf.tables_initializer(),
        *tf.get_collection(tf.GraphKeys.LOCAL_INIT_OP))
    return tf.train.Scaffold(local_init_op=local_init_op)


def train(hparams, dataset, log_dir, num_steps, valid_dataset=None):
    """Trains/continues training a VAE and saves it in log_dir."""
    vae = make_vae(hparams)
    observed = observations(dataset)
    train_op, debug_tensors = train_graph(hparams, vae, observed)
    eval_graph(hparams, vae, observed, name="train_eval")
    if valid_dataset is not None:
        eval_observed = observations(valid_dataset)
        eval_graph(hparams, vae, eval_observed, name="valid_eval")
    generated = gen_graph(hparams, vae)
    train_midi = tf.summary.audio(
        "observed",
        synthesize(hparams, observed),
        SYNTHESIZED_RATE,
        collections=[])
    gen_midi = tf.summary.audio(
        "generated",
        synthesize(hparams, generated),
        SYNTHESIZED_RATE,
        collections=[])
    display_hook = tf.train.SummarySaverHook(
        save_steps=1000, output_dir=log_dir,
        summary_op=tf.summary.merge([train_midi, gen_midi]))
    logging_hook = tf.train.LoggingTensorHook(debug_tensors, every_n_secs=60.)
    hooks = [logging_hook, display_hook]
    metric_updates = tf.get_collection("metric_updates")
    with tf.train.MonitoredTrainingSession(scaffold=make_scaffold(),
                                           checkpoint_dir=log_dir,
                                           is_chief=True,
                                           hooks=hooks) as sess:
        for i in range(num_steps):
            if sess.should_stop():
                break
            ops = [train_op]
            if i % 100 == 0:
                ops.extend(metric_updates)
            sess.run(ops)


def evaluate(hparams, dataset, log_dir, num_steps):
    """Calculates the mean log-prob for the given sequences."""
    observed = observations(dataset)
    vae = make_vae(hparams)
    mean_log_prob = eval_graph(hparams, vae, observed)
    logging_hook = tf.train.LoggingTensorHook(
        [mean_log_prob], every_n_secs=10., at_end=True)
    with tf.train.MonitoredSession(
        hooks=[logging_hook],
        session_creator=tf.train.ChiefSessionCreator(
            scaffold=make_scaffold(),
            checkpoint_dir=log_dir)) as sess:
        for _ in range(num_steps):
            sess.run(tf.get_collection("metric_updates"))


def generate(hparams, log_dir, out_dir, num_samples):
    """Prints out strings generated from the model."""
    vae = make_vae(hparams)
    generated = gen_graph(hparams, vae)
    synthesized = synthesize(hparams, generated)
    with tf.train.MonitoredSession(
        session_creator=tf.train.ChiefSessionCreator(
            scaffold=make_scaffold(),
            checkpoint_dir=log_dir)) as sess:
        num = 0
        while True:
            for sample in sess.run(synthesized):
                if num >= num_samples:
                    return
                basename = "generated_{:02}.wav".format(num)
                out_path = os.path.join(out_dir, basename)
                scipy.io.wavfile.write(out_path, SYNTHESIZED_RATE, sample)
                print("Wrote " + out_path)
                num += 1
