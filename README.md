# VAE-Seq

VAE-Seq is a library for modeling sequences of observations.

## Framing

Lets define some terminology to frame the problem. We'd like to model
a sequence of *observations*. An observation could be a frame in a
game being played, a window of samples from an audio file, or the
price of a stock at a certain time.

Our model for the probability of an observation at a given depends on:

  * *Latent variables*, both deterministic and stochastic.
  * A *context* that encodes input from the environment.

The latent variables depend on the generative model, but generally
they are composed of a deterministic RNN state and a set stochastic
variables drawn from some easy-to-sample probability distribution.

The context usually encodes information about the previous
observations, as well as any external inputs. For example, when
modeling a game, the context might be an encoding of the last few
frames and the last action that was played.

## Components

The core of the model is one of the VAE implementations defined in
vae_seq/vae. To construct it, you must supply:

  * An *Agent* (see: `vae_seq/agent.py`) that receives observations
    and supplies contexts.
  * An *Encoder*, a Sonnet module that takes an observation and
    outputs a flat representation.
  * A *Decoder*, a `DistModule` (see: `vae_seq/dist_module.py`) that
    produces probability distributions over observations.

## Generative Models

We implement three generative models (see: `vae_seq/vae`):

  * An RNN with no latent variables other than a deterministic state.
  * A VAE where the stochastic latent variables are independent.
  * An implementation of SRNN (https://arxiv.org/abs/1605.07571)

To train these models, you may use `TrainOps` (see:
`vae_seq/train.py`). It accepts a sequences of contexts and
observations and returns an operation that must be run repeatedly to
train the model.

Once trained, you can sample sequences by running the model's
*GenCore* (see: `vae_seq/vae/base.py`), which alternates between
running the agent to get a context and sampling from the generative
model to get an observation.


## Examples

### Toy Game

In `examples/toy_game`, we model a (very) simple partially observable
game environment. See `examples/toy_game/game.py` for the rules, but
it's not a very fun game.

The agent here takes random moves in the game during
training; we wrap the game implementation in an `agent.Environment` so
that we can use `agent.contexts_and_observations_from_environment` to
generate the training data.

To train, run:
```shell
$ bazel run -c opt //examples/toy_game:train -- \
    --log_dir /tmp/toy_game \
    --iters 10000 \
    --hparams "vae_type=SRNN"
```

Then you can play the modeled game:
```shell
$ bazel run -c opt //examples/toy_game:play -- \
    --log_dir /tmp/toy_game \
    --hparams "vae_type=SRNN"  # make sure to keep the same HPARAMS
```

### MusicNet

In `examples/music_net`, we model audio samples from the MusicNet
dataset (see: http://homes.cs.washington.edu/~thickstn/musicnet.html).

Here, we have a static set of observations (unlike in the toy game,
where our training data depends on the agent's actions). Observations
are sequences of 200 samples at 16000 HZ audio rate.

The agent context is just an encoding of the previous observation, and
we use `agent.contexts_for_static_observations` to generate the
contexts for training.

