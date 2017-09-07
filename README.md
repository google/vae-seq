# VAE-Seq

VAE-Seq is a library for modeling sequences of observations.

## Background

One tool that's commonly used to model sequential data is the
Recurrent Neural Network (RNN), or gated variations of it such as the
Long Short-Term Memory cell or the Gated Recurrent Unit cell.

RNNs in general are essentially trainable transition functions:
`(input, state) -> (output, state')`, and by themselves don't specify
a complete model. We additionally need to specify a family of
distributions that describes our observations; common choices here are
`Categorical` distributions for discrete observations such as text or
`Normal` distributions for real-valued observations.

The `output` of the RNN specifies the parameters of the observation
distribution (e.g. the logits of a `Categorical` or the mean and
variance of a `Normal`). But the size of the RNN `output` and the
number of parameters that we need don't necessarily match up. To solve
this, we project `output` into the appropriate shape via a Neural
Network we'll call a decoder.

And what about the `input` of the RNN? It can be empty, but we might
want to include side information from the environment (e.g. actions
when modeling a game or a metronome when modeling
music). Additionally, the observation from the previous step(s) is
almost always an important feature to include. Here, we'll use another
Neural Network we'll call an encoder to summarize the observation
into a more digestible form.

Together, these components specify a factored (by time step)
probability distribution that we can train in the usual way: by
maximizing the probability of the network weights given the
observations in your training data and your priors over those
weights. Once trained, you can use ancestral sampling to generate new
sequences.

## Motivation

This library allows you to express the type of model described
above. It handles the plumbing for you: you define the encoder, the
decoder, and the observation distribution. The resulting model can
be trained on a `Dataset` of observation sequences, queried for the
probability of a given sequence, or queried to generate new sequences.

But the model above also has a limitation: the family of observation
distributions we pick is the only source of non-determinism in the
model. If it can't express the true distribution of observations, the
model won't be able to learn or generate the true range of observation
sequences. For example, consider a sequence of black/white images. If
we pick the observation distribution to be a set of independent
`Bernoulli` distributions over pixel values, the first generated image
would always look like a noisy average over images in the training
set. Subsequent images might get more creative since they are
conditioned on a noisy input, but that depends on how images vary
between steps in the training data.

The issue in the example above is that the observation distribution we
picked wasn't expressive enough: pixels in an image aren't
independent. One way to fix this is to design very expressive
observation distributions that can model images. Another way is to
condition the simple distribution on a latent variable to produce a
hierarchical output distribution. This latter type of model is known
as a Variational Auto encoder (VAE).

There are different ways to incorporate latent variables in a
sequential model (see the supported architectures below) but the
general approach we take here is to view the RNN `state` as a
collection of stochastic and deterministic variables.

## Usage

To define a model, subclass `ModelBase` to define an encoder, a
decoder, and the output distribution. The decoder and output
distribution are packaged together into a `DistModule` (see:
[vaeseq/codec.py](vaeseq/codec.py)).

The following model architectures are currently available (see:
[vaeseq/vae](vaeseq/vae)):

  * An RNN with no latent variables other than a deterministic state.
  * A VAE where the stochastic latent variables are independent across
    time steps.
  * An implementation of SRNN (https://arxiv.org/abs/1605.07571)

There are lots of hyper-parameters packaged into an `HParams` object
(see: [vaeseq/hparams.py](vaeseq/hparams.py)). You can select among
the architectures above by setting the `vae_type` parameter.

## Examples

When you build and install this library via `python setup.py install`,
the following example programs are installed as well. See:
[vaeseq/examples](vaeseq/examples).

### Text

A character-sequence model that can be used to generate nonsense text
or to evaluate the probability that a given piece of text was written
by a given author.

To train on Andrej Karpathy's "Tiny Shakespeare" dataset:
```shell
$ wget https://github.com/karpathy/char-rnn/raw/master/data/tinyshakespeare/input.txt
$ vaeseq-text train --log-dir /tmp/text --train-corpus input.txt \
    --num-steps 1000000
```

After training has completed, you can generate text:
```shell
$ vaeseq-text generate --log-dir /tmp/text --vocab-corpus input.txt \
    --length 1000
    --num-samples 20
```

Or you can tell how likely a piece of text is to be Shakespearean:
```shell
$ vaeseq-text evaluate --log-dir /tmp/text --vocab-corpus input.txt \
    --eval-corpus foo.txt
```

### MIDI

Similar to the text example above, but now modeling MIDI music
(specifically, piano rolls). Installed under `vaeseq-midi`. Don't
expect it to sound great.

### Play

An experiment modeling a game environment and using that to train an
agent via policy gradient. This example uses the OpenAI Gym
module. Installed under `vaeseq-play`.

## Disclaimer

This is not an official Google product.
