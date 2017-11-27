"""Basic tests for all of the VAE implementations."""

import tensorflow as tf
import sonnet as snt

from vaeseq import codec
from vaeseq import context as context_mod
from vaeseq import hparams as hparams_mod
from vaeseq import util
from vaeseq import vae as vae_mod


def _context_and_vae(hparams):
    """Constructs a VAE."""
    obs_encoder = codec.MLPObsEncoder(hparams)
    obs_decoder = codec.MLPObsDecoder(
        hparams,
        codec.BernoulliDecoder(squeeze_input=True),
        param_size=1)
    context = context_mod.EncodeObserved(obs_encoder)
    vae = vae_mod.make(hparams, obs_encoder, obs_decoder)
    return context, vae


def _observed(hparams):
    """Test observations."""
    return tf.zeros([util.batch_size(hparams), util.sequence_size(hparams)],
                    dtype=tf.int32, name="test_obs")


def _inf_tensors(hparams, context, vae):
    """Simple inference graph."""
    with tf.name_scope("inf"):
        observed = _observed(hparams)
        contexts = context.from_observations(inputs=None, observed=observed)
        latents, divs = vae.infer_latents(contexts, observed)
        log_probs = vae.evaluate(contexts, observed, latents=latents)
        elbo = tf.reduce_sum(log_probs - divs)
    return [observed, latents, divs, log_probs, elbo]


def _gen_tensors(hparams, context, vae):
    """Samples observations and latent variables from the VAE."""
    del hparams  # Unused, just passed for consistency.
    with tf.name_scope("gen"):
        generated, latents = vae.generate(inputs=None, context=context)
    return [generated, latents]


def _eval_tensors(hparams, context, vae):
    """Calculates the log-probabilities of the observations."""
    with tf.name_scope("eval"):
        observed = _observed(hparams)
        contexts = context.from_observations(inputs=None, observed=observed)
        log_probs = vae.evaluate(contexts, observed, samples=100)
    return [log_probs]


def _test_assertions(inf_tensors, gen_tensors, eval_tensors):
    """Returns in-graph assertions for testing."""
    observed, latents, divs, log_probs, elbo = inf_tensors
    generated, sampled_latents = gen_tensors
    eval_log_probs, = eval_tensors

    # For RNN, we return None from infer_latents as an optimization.
    if latents is None:
        latents = sampled_latents

    def _same_batch_and_sequence_size_asserts(t1, name1, t2, name2):
        return [
            tf.assert_equal(
                util.batch_size_from_nested_tensors(t1),
                util.batch_size_from_nested_tensors(t2),
                message="Batch: " + name1 + " vs " + name2),
            tf.assert_equal(
                util.sequence_size_from_nested_tensors(t1),
                util.sequence_size_from_nested_tensors(t2),
                message="Steps: " + name1 + " vs " + name2),
        ]

    def _same_shapes(nested1, nested2):
        return snt.nest.flatten(snt.nest.map(
            lambda t1, t2: tf.assert_equal(
                tf.shape(t1), tf.shape(t2),
                message="Shapes: " + t1.name + " vs " + t2.name),
            nested1, nested2))

    def _all_same_batch_and_sequence_sizes(nested):
        batch_size = util.batch_size_from_nested_tensors(nested)
        sequence_size = util.sequence_size_from_nested_tensors(nested)
        return [
            tf.assert_equal(tf.shape(tensor)[0], batch_size,
                            message="Batch: " + tensor.name)
            for tensor in snt.nest.flatten(nested)
        ] + [
            tf.assert_equal(tf.shape(tensor)[1], sequence_size,
                            message="Steps: " + tensor.name)
            for tensor in snt.nest.flatten(nested)
        ]

    assertions = [
        tf.assert_non_negative(divs),
        tf.assert_non_positive(log_probs),
    ] + _same_shapes(
        (log_probs, log_probs,      observed,  latents),
        (divs,      eval_log_probs, generated, sampled_latents)
    ) + _all_same_batch_and_sequence_sizes(
        (observed, latents, divs)
    ) + _all_same_batch_and_sequence_sizes(
        (generated, sampled_latents)
    )
    vars_ = tf.trainable_variables()
    grads = tf.gradients(-elbo, vars_)
    for (var, grad) in zip(vars_, grads):
        assertions.append(tf.check_numerics(grad, "Gradient for " + var.name))
    return assertions


def _all_tensors(hparams, context, vae):
    """All tensors to evaluate in tests."""
    gen_tensors = _gen_tensors(hparams, context, vae)
    inf_tensors = _inf_tensors(hparams, context, vae)
    eval_tensors = _eval_tensors(hparams, context, vae)
    assertions = _test_assertions(inf_tensors, gen_tensors, eval_tensors)
    all_tensors = inf_tensors + gen_tensors + eval_tensors + assertions
    return [x for x in all_tensors if x is not None]


class VAETest(tf.test.TestCase):

    def _test_vae(self, vae_type):
        """Make sure that all tensors and assertions evaluate without error."""
        hparams = hparams_mod.make_hparams(vae_type=vae_type)
        context, vae = _context_and_vae(hparams)
        tensors = _all_tensors(hparams, context, vae)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tensors)

    def test_iseq(self):
        self._test_vae("ISEQ")

    def test_rnn(self):
        self._test_vae("RNN")

    def test_srnn(self):
        self._test_vae("SRNN")


if __name__ == "__main__":
    tf.test.main()
