package(default_visibility = ["//visibility:public"])

py_library(
    name = "independent_sequence",
    srcs = ["independent_sequence.py"],
    deps = [
        "//vaeseq:latent",
        "//vaeseq:util",
        "//vaeseq:vae_module",
    ],
)

py_library(
    name = "rnn",
    srcs = ["rnn.py"],
    deps = [
        "//vaeseq:util",
        "//vaeseq:vae_module",
    ],
)

py_library(
    name = "srnn",
    srcs = ["srnn.py"],
    deps = [
        "//vaeseq:latent",
        "//vaeseq:util",
        "//vaeseq:vae_module",
    ],
)

py_library(
    name = "vae",
    srcs = ["__init__.py"],
    deps = [
        ":independent_sequence",
        ":rnn",
        ":srnn",
    ],
)

py_test(
    name = "vae_test",
    srcs = ["vae_test.py"],
    deps = [
        ":vae",
        "//vaeseq:codec",
        "//vaeseq:context",
        "//vaeseq:hparams",
        "//vaeseq:util",
    ],
)
