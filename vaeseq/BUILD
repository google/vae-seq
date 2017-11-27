# Everything depends on TensorFlow and Sonnet.
package(default_visibility = ["//visibility:public"])

test_suite(
    name = "tests",
    tests = [
        "batch_dist_test",
        "context_test",
        "model_test",
        "util_test",
        "//vaeseq/examples/midi:tests",
        "//vaeseq/examples/play:tests",
        "//vaeseq/examples/text:tests",
        "//vaeseq/vae:vae_test",
    ],
)

py_library(
    name = "dist_module",
    srcs = ["dist_module.py"],
    deps = [":util"],
)

py_library(
    name = "vae_module",
    srcs = ["vae_module.py"],
    deps = [
        ":dist_module",
        ":util",
    ],
)

py_library(
    name = "batch_dist",
    srcs = ["batch_dist.py"],
)

py_test(
    name = "batch_dist_test",
    srcs = ["batch_dist_test.py"],
    deps = [
        ":batch_dist",
    ],
)

py_library(
    name = "latent",
    srcs = ["latent.py"],
    deps = [
        ":dist_module",
        ":util",
    ],
)

py_library(
    name = "hparams",
    srcs = ["hparams.py"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":hparams",
        ":util",
    ],
)

py_library(
    name = "context",
    srcs = ["context.py"],
    deps = [
        ":util",
    ],
)

py_test(
    name = "context_test",
    srcs = ["context_test.py"],
    deps = [
        ":codec",
        ":context",
    ],
)

py_library(
    name = "codec",
    srcs = ["codec.py"],
    deps = [
        ":batch_dist",
        ":dist_module",
        ":util",
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":util",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":context",
        ":train",
        ":util",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":codec",
        ":hparams",
        ":model",
        ":util",
        "//vaeseq/vae",
    ],
)
