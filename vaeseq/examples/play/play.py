# Copyright 2018 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model Cart-Pole and train an Agent via policy gradient."""

import argparse
import time
import sys
import tensorflow as tf

from vaeseq.examples.play import environment as env_mod
from vaeseq.examples.play import hparams as hparams_mod
from vaeseq.examples.play import model as model_mod
from vaeseq import util


def train(flags):
    model = model_mod.Model(
        hparams=hparams_mod.make_hparams(flags.hparams),
        session_params=flags)
    model.train("train", flags.num_steps)

def run(flags):
    hparams = hparams_mod.make_hparams(flags.hparams)
    hparams.batch_size = 1
    hparams.sequence_size = flags.max_moves
    batch_size = util.batch_size(hparams)
    model = model_mod.Model(hparams=hparams, session_params=flags)
    if flags.agent == "trained":
        agent = model.agent
    elif flags.agent == "random":
        agent = model_mod.agent_mod.RandomAgent(hparams)
    else:
        raise ValueError("I don't understand --agent " + flags.agent)
    outputs = agent.drive_rnn(
        model.env,
        sequence_size=util.sequence_size(hparams),
        initial_state=agent.initial_state(batch_size=batch_size),
        cell_initial_state=model.env.initial_state(batch_size=batch_size))
    score = tf.reduce_sum(outputs["score"])
    with model.eval_session() as sess:
        model.env.start_render_thread()
        for _ in range(flags.num_games):
            print("Score: ", sess.run(score))
            sys.stdout.flush()
        model.env.stop_render_thread()


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


def run_args(args):
    common_args(args)
    args.add_argument(
        "--max-moves", type=int, default=1000,
        help="Maximum number of moves per game.")
    args.add_argument(
        "--num-games", type=int, default=1,
        help="Number of games to play.")
    args.add_argument(
        "--agent", default="trained", choices=["trained", "random"],
        help="Which agent to use.")
    args.set_defaults(entry=run)


def main():
    args = argparse.ArgumentParser()
    subcommands = args.add_subparsers(title="subcommands")
    train_args(subcommands.add_parser(
        "train", help="Train a model."))
    run_args(subcommands.add_parser(
        "run", help="Run a traned model."))
    flags, unparsed_args = args.parse_known_args(sys.argv[1:])
    if not hasattr(flags, "entry"):
        args.print_help()
        return 1
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=lambda _unused_argv: flags.entry(flags),
               argv=[sys.argv[0]] + unparsed_args)


if __name__ == "__main__":
    main()
