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
vaeseq/vae. To construct it, you must supply:

  * An *Agent* (see: `vaeseq/agent.py`) that receives observations
    and supplies contexts.
  * An *Encoder*, a Sonnet module that takes an observation and
    outputs a flat representation.
  * A *Decoder*, a `DistModule` (see: `vaeseq/dist_module.py`) that
    produces probability distributions over observations.

## Generative Models

We implement three generative models (see: `vaeseq/vae`):

  * An RNN with no latent variables other than a deterministic state.
  * A VAE where the stochastic latent variables are independent.
  * An implementation of SRNN (https://arxiv.org/abs/1605.07571)

To train these models, you may use `TrainOps` (see:
`vaeseq/train.py`). It accepts a sequences of contexts and
observations and returns an operation that must be run repeatedly to
train the model.

Once trained, you can sample sequences by running the model's
*GenCore* (see: `vaeseq/vae/base.py`), which alternates between
running the agent to get a context and sampling from the generative
model to get an observation.


## Examples

### Text

A character-sequence model that can be used to generate nonsense text
or to evaluate the probability that a given piece of text was written
by a given author.

To train on Andrej Karpathy's "Tiny Shakespeare" dataset:
```shell
$ wget https://github.com/karpathy/char-rnn/raw/master/data/tinyshakespeare/input.txt
$ bazel run -c opt //vaeseq/examples/text:train -- \
    --train_corpus input.txt \
    --log_dir /tmp/text \
    --hparams "vae_type=SRNN,sequence_size=40" \
    --iters 1000000
```

After training has completed, you can generate text:
```shell
$ bazel run -c opt //vaeseq/examples/text:generate -- \
    --train_corpus input.txt \
    --log_dir /tmp/text \
    --hparams "vae_type=SRNN,sequence_size=1000" \
    --samples 20
```
