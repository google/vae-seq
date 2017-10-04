"""Functions to build up training and generation graphs."""

import tensorflow as tf

from vae_seq import agent as agent_mod
from vae_seq import train as train_mod
from vae_seq import util
from vae_seq import vae as vae_mod

from . import codec


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
        contexts = agent_mod.contexts_for_static_observations(
            observed,
            vae.agent,
            agent_inputs)
    train_ops = train_mod.TrainOps(hparams, vae)
    train_op, debug_tensors = train_ops(contexts, observed)
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


def observations_to_strings(observed, id_to_char):
    """Returns a batch of strings corresponding to the observation sequences."""
    # Note, tf.reduce_sum doesn't work on strings.
    chars = id_to_char.lookup(tf.to_int64(observed))
    return tf.py_func(lambda chars: chars.sum(axis=1), [chars], [tf.string])[0]


def observations(dataset, char_to_id):
    """Returns a sequence of observations (IDs) from the dataset."""
    iterator = tf.contrib.data.Iterator.from_dataset(dataset)
    tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, iterator.initializer)
    chars = iterator.get_next()
    return char_to_id.lookup(chars)


def make_scaffold():
    local_init_op = tf.group(
        tf.local_variables_initializer(),
        tf.tables_initializer(),
        *tf.get_collection(tf.GraphKeys.LOCAL_INIT_OP))
    return tf.train.Scaffold(local_init_op=local_init_op)


def train(hparams, dataset, char_to_id, id_to_char, log_dir, num_steps,
          valid_dataset=None):
    """Trains/continues training a VAE and saves it in log_dir."""
    vae = make_vae(hparams)
    observed = observations(dataset, char_to_id)
    train_op, debug_tensors = train_graph(hparams, vae, observed)
    eval_graph(hparams, vae, observed, name="train_eval")
    if valid_dataset is not None:
        eval_observed = observations(valid_dataset, char_to_id)
        eval_graph(hparams, vae, eval_observed, name="valid_eval")
    generated = gen_graph(hparams, vae)
    train_text = tf.summary.text(
        "observed",
        observations_to_strings(observed, id_to_char),
        collections=[])
    gen_text = tf.summary.text(
        "generated",
        observations_to_strings(generated, id_to_char),
        collections=[])
    display_hook = tf.train.SummarySaverHook(
        save_steps=1000, output_dir=log_dir,
        summary_op=tf.summary.merge([train_text, gen_text]))
    logging_hook = tf.train.LoggingTensorHook(debug_tensors, every_n_secs=60.)
    hooks = [logging_hook, display_hook]
    metric_updates = tf.get_collection("metric_updates")
    with tf.train.MonitoredTrainingSession(scaffold=make_scaffold(),
                                           checkpoint_dir=log_dir,
                                           is_chief=True,
                                           hooks=hooks) as sess:
        for i in xrange(num_steps):
            if sess.should_stop():
                break
            ops = [train_op]
            if i % 100 == 0:
                ops.extend(metric_updates)
            sess.run(ops)


def evaluate(hparams, dataset, char_to_id, log_dir, num_steps):
    """Calculates the mean log-prob for the given sequences."""
    observed = observations(dataset, char_to_id)
    vae = make_vae(hparams)
    mean_log_prob = eval_graph(hparams, vae, observed)
    logging_hook = tf.train.LoggingTensorHook(
        [mean_log_prob], every_n_secs=10., at_end=True)
    with tf.train.MonitoredSession(
        hooks=[logging_hook],
        session_creator=tf.train.ChiefSessionCreator(
            scaffold=make_scaffold(),
            checkpoint_dir=log_dir)) as sess:
        for _ in xrange(num_steps):
            sess.run(tf.get_collection("metric_updates"))


def generate(hparams, id_to_char, log_dir, num_samples):
    """Prints out strings generated from the model."""
    vae = make_vae(hparams)
    generated = gen_graph(hparams, vae)
    gen_text = observations_to_strings(generated, id_to_char)
    with tf.train.MonitoredSession(
        session_creator=tf.train.ChiefSessionCreator(
            scaffold=make_scaffold(),
            checkpoint_dir=log_dir)) as sess:
        num = 0
        while num < num_samples:
            batch = sess.run(gen_text)[:num_samples - num]
            for sample in batch:
                print(sample)
                print("===========")
            num += len(batch)
