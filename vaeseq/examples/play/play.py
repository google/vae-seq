"""Model Cart-Pole and train an Agent via policy gradient."""

import argparse
import sys
import tensorflow as tf

from vaeseq.examples.play import hparams as hparams_mod
from vaeseq.examples.play import model as model_mod


def train(flags):
    model = model_mod.Model(
        hparams=hparams_mod.make_hparams(flags.hparams),
        session_params=flags)
    model.train("train", flags.num_steps)


# Argument parsing code below.

def common_args(args):
    model_mod.Model.SessionParams.add_parser_arguments(args)
    args.add_argument(
        "--hparams", default="",
        help="Model hyperparameter overrides.")


def train_args(args):
    common_args(args)
    args.add_argument(
        "--num-steps", type=int, default=int(1e6),
        help="Number of training iterations.")
    args.set_defaults(entry=train)


def main():
    args = argparse.ArgumentParser()
    subcommands = args.add_subparsers(title="subcommands")
    train_args(subcommands.add_parser(
        "train", help="Train a model."))
    flags, unparsed_args = args.parse_known_args(sys.argv[1:])
    if not hasattr(flags, "entry"):
        args.print_help()
        return 1
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=lambda _unused_argv: flags.entry(flags),
               argv=[sys.argv[0]] + unparsed_args)


if __name__ == "__main__":
    main()
