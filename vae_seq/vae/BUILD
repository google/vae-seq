package(default_visibility = ["//visibility:public"])

py_library(
    name = "latent",
    srcs = ["latent.py"],
    deps = [
        "//vae_seq:util",
    ],
)

py_library(
    name = "independent_sequence",
    srcs = ["independent_sequence.py"],
    deps = [
        ":latent",
        "//vae_seq:feedback_rnn_core",
        "//vae_seq:util",
    ],
)

py_library(
    name = "srnn",
    srcs = ["srnn.py"],
    deps = [
        ":latent",
        "//vae_seq:feedback_rnn_core",
        "//vae_seq:util",
    ],
)

py_library(
    name = "vae",
    srcs = ["__init__.py"],
    deps = [
        ":independent_sequence",
        ":srnn",
    ],
)
