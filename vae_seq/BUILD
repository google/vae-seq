# Everything depends on TensorFlow and Sonnet.
package(default_visibility = ["//visibility:public"])

py_library(
    name = "feedback_rnn_core",
    srcs = ["feedback_rnn_core.py"],
)

py_library(
    name = "hparams",
    srcs = ["hparams.py"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
)

py_library(
    name = "obs_layers",
    srcs = ["obs_layers.py"],
    deps = [
        ":util",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//vae_seq/vae",
    ],
)