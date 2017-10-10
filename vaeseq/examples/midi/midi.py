"""Model MIDI sequences."""

import argparse
import itertools
import os.path
import sys

import scipy.io.wavfile
import tensorflow as tf

from vaeseq.examples.midi import hparams as hparams_mod
from vaeseq.examples.midi import model as model_mod


def train(flags):
    model = model_mod.Model(
        hparams=hparams_mod.make_hparams(flags.hparams),
        session_params=flags)
    model.train(flags.train_files, flags.num_steps,
                valid_dataset=flags.valid_files)


def evaluate(flags):
    model = model_mod.Model(
        hparams=hparams_mod.make_hparams(flags.hparams),
        session_params=flags)
    model.evaluate(flags.eval_files, flags.num_steps)


def generate(flags):
    hparams = hparams_mod.make_hparams(flags.hparams)
    hparams.sequence_size = int(hparams.rate * flags.length)
    model = model_mod.Model(hparams=hparams, session_params=flags)
    samples = itertools.islice(model.generate(), flags.num_samples)
    for i, wav in enumerate(samples):
        basename = "generated_{:02}.wav".format(i + 1)
        tf.logging.info("Writing %s.", basename)
        out_path = os.path.join(flags.out_dir, basename)
        scipy.io.wavfile.write(out_path, model_mod.Model.SYNTHESIZED_RATE, wav)


# Argument parsing code below.

def common_args(args):
    model_mod.Model.SessionParams.add_parser_arguments(args)
    args.add_argument(
        "--hparams", default="",
        help="Model hyperparameter overrides.")


def train_args(args):
    common_args(args)
    args.add_argument(
        "--train-files", nargs="+",
        help="MIDI files to train on.",
        required=True)
    args.add_argument(
        "--valid-files", nargs="+",
        help="MIDI files to evaluate while training.")
    args.add_argument(
        "--num-steps", type=int, default=int(1e6),
        help="Number of training iterations.")
    args.set_defaults(entry=train)


def eval_args(args):
    common_args(args)
    args.add_argument(
        "--eval-files", nargs="+",
        help="MIDI files to evaluate.",
        required=True)
    args.add_argument(
        "--num-steps", type=int, default=int(1e3),
        help="Number of eval iterations.")
    args.set_defaults(entry=evaluate)


def generate_args(args):
    common_args(args)
    args.add_argument(
        "--out-dir",
        help="Where to store the generated sequences.",
        required=True)
    args.add_argument(
        "--length", type=float, default=5.,
        help="Length of the generated sequences, in seconds.")
    args.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of sequences to generate.")
    args.set_defaults(entry=generate)


def main():
    args = argparse.ArgumentParser()
    subcommands = args.add_subparsers(title="subcommands")
    train_args(subcommands.add_parser(
        "train", help="Train a model."))
    eval_args(subcommands.add_parser(
        "evaluate", help="Evaluate a trained model."))
    generate_args(subcommands.add_parser(
        "generate", help="Generate some music."))
    flags, unparsed_args = args.parse_known_args(sys.argv[1:])
    if not hasattr(flags, "entry"):
        args.print_help()
        return 1
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=lambda _unused_argv: flags.entry(flags),
               argv=[sys.argv[0]] + unparsed_args)


if __name__ == "__main__":
    main()
