"""Model sequences of text, character-by-character."""

from __future__ import print_function

import argparse
import itertools
import sys
import tensorflow as tf

from vaeseq.examples.text import hparams as hparams_mod
from vaeseq.examples.text import model as model_mod


def train(flags):
    if flags.vocab_corpus is None:
        print("NOTE: no --vocab-corpus supplied; using",
              repr(flags.train_corpus), "for vocabulary.")
    model = model_mod.Model(
        hparams=hparams_mod.make_hparams(flags.hparams),
        session_params=flags,
        vocab_corpus=flags.vocab_corpus or flags.train_corpus)
    model.train(flags.train_corpus, flags.num_steps,
                valid_dataset=flags.valid_corpus)


def evaluate(flags):
    model = model_mod.Model(
        hparams=hparams_mod.make_hparams(flags.hparams),
        session_params=flags,
        vocab_corpus=flags.vocab_corpus)
    model.evaluate(flags.eval_corpus, flags.num_steps)


def generate(flags):
    hparams = hparams_mod.make_hparams(flags.hparams)
    hparams.sequence_size = flags.length
    model = model_mod.Model(
        hparams=hparams,
        session_params=flags,
        vocab_corpus=flags.vocab_corpus)
    for i, string in enumerate(itertools.islice(model.generate(),
                                                flags.num_samples)):
        print("#{:02d}: {}\n".format(i + 1, string))


# Argument parsing code below.

def common_args(args, require_vocab):
    model_mod.Model.SessionParams.add_parser_arguments(args)
    args.add_argument(
        "--hparams", default="",
        help="Model hyperparameter overrides.")
    args.add_argument(
        "--vocab-corpus",
        help="Path to the corpus used for vocabulary generation.",
        required=require_vocab)


def train_args(args):
    common_args(args, require_vocab=False)
    args.add_argument(
        "--train-corpus",
        help="Location of the training text.",
        required=True)
    args.add_argument(
        "--valid-corpus",
        help="Location of the validation text.")
    args.add_argument(
        "--num-steps", type=int, default=int(1e6),
        help="Number of training iterations.")
    args.set_defaults(entry=train)


def eval_args(args):
    common_args(args, require_vocab=True)
    args.add_argument(
        "--eval-corpus",
        help="Location of the training text.",
        required=True)
    args.add_argument(
        "--num-steps", type=int, default=int(1e3),
        help="Number of eval iterations.")
    args.set_defaults(entry=evaluate)


def generate_args(args):
    common_args(args, require_vocab=True)
    args.add_argument(
        "--length", type=int, default=1000,
        help="Length of the generated strings.")
    args.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of strings to generate.")
    args.set_defaults(entry=generate)


def main():
    args = argparse.ArgumentParser()
    subcommands = args.add_subparsers(title="subcommands")
    train_args(subcommands.add_parser(
        "train", help="Train a model."))
    eval_args(subcommands.add_parser(
        "evaluate", help="Evaluate a trained model."))
    generate_args(subcommands.add_parser(
        "generate", help="Generate some text."))
    flags, unparsed_args = args.parse_known_args(sys.argv[1:])
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=lambda _unused_argv: flags.entry(flags),
               argv=[sys.argv[0]] + unparsed_args)


if __name__ == "__main__":
    main()
