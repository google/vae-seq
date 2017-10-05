"""Functions to build up training and generation graphs."""

import tensorflow as tf

from vae_seq import codec
from vae_seq import train as train_mod
from vae_seq import vae as vae_mod

from . import agent as agent_mod
from . import environment as env_mod
from . import game as game_mod


def display(actions, latents, observed):
    """Returns a markdown-formatted description of the sequence."""
    lines = []
    zip_b = zip(actions, latents, observed)
    for actions_b, latents_b, observed_b in zip_b[:5]:  # iterate over batches
        zip_t = zip(actions_b, latents_b, observed_b)
        for action_t, latent_t, observed_t in zip_t:  # iterate over steps
            lines.append("    ACTION:      " + game_mod.ACTIONS[action_t])
            lines.append("    OBSERVATION: " + repr(list(observed_t)))
            lines.append("    LATENT:      " + repr(list(latent_t)))
            lines.append("    ")
        lines.append("----\n")
    return '\n'.join(lines)


def make_vae(hparams):
    """Constructs a VAE for modeling test games."""
    with tf.name_scope("vae"):
        obs_encoder = codec.MLPObsEncoder(hparams)
        obs_decoder = codec.OneHotObsDecoder(hparams)
        agent = agent_mod.Agent(hparams, obs_encoder)
        return vae_mod.make(hparams, agent, obs_encoder, obs_decoder)


def train_graph(hparams, vae):
    """Constructs a training graph."""
    with tf.name_scope("training_data"):
        contexts, observations = vae.agent.contexts_and_observations(
            env_mod.Environment(hparams))
    train_ops = train_mod.TrainOps(hparams, vae)
    train_op, debug_tensors = train_ops(contexts, observations)
    return train_op, debug_tensors


def gen_graph(hparams, vae):
    """Constructs a generation graph."""
    del hparams  # not used, only passed in for consistency.
    with tf.name_scope("generate"):
        agent_inputs = vae.agent.inputs()
        generated, latents, agent_states = vae.gen_core.generate(agent_inputs)
        env_inputs = tf.map_fn(
            lambda args: vae.agent.env_input(*args),
            (agent_inputs, agent_states),
            dtype=tf.int32)
        return env_inputs, latents, generated


def train(hparams, log_dir, num_steps):
    """Trains/continues training a VAE and saves it in log_dir."""
    vae = make_vae(hparams)
    train_op, debug_tensors = train_graph(hparams, vae)
    env_inputs, latents, generated = gen_graph(hparams, vae)
    debug_tensors["step"] = tf.train.get_or_create_global_step()
    logging_hook = tf.train.LoggingTensorHook(debug_tensors, every_n_secs=60.)
    display_summary = tf.summary.text(
        "display",
        tf.py_func(display, [env_inputs, latents, generated], [tf.string])[0],
        collections=[])
    display_hook = tf.train.SummarySaverHook(
        save_steps=1000, output_dir=log_dir, summary_op=display_summary)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=log_dir,
        is_chief=True,
        hooks=[logging_hook, display_hook]) as sess:
        for _ in six.range(num_steps):
            if sess.should_stop():
                break
            sess.run(train_op)


def play(hparams, log_dir):
    """Loads a trained VAE and plays interactively against it."""
    hparams.batch_size = 1
    vae = make_vae(hparams)
    vae.agent.interactive = True
    _unused_env_inputs, _unused_latents, generated = gen_graph(hparams, vae)
    with tf.train.MonitoredSession(
        session_creator=tf.train.ChiefSessionCreator(
            checkpoint_dir=log_dir)) as sess:
        sess.run(generated)
