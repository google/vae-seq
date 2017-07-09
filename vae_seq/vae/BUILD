package(default_visibility = ["//visibility:public"])

py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [],
)

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
        ":base",
        ":latent",
        ":independent_sequence_distribution",
        "//vae_seq:util",
    ],
)

py_library(
    name = "independent_sequence_distribution",
    srcs = ["independent_sequence_distribution.py"],
    deps = [],
)

py_library(
    name = "srnn",
    srcs = ["srnn.py"],
    deps = [
        ":base",
        ":latent",
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

py_test(
    name = "vae_test",
    srcs = ["vae_test.py"],
    deps = [
        ":vae",
        "//vae_seq:hparams",
        "//vae_seq:obs_layers",
        "//vae_seq:util",
    ],
)
