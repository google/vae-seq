package(default_visibility = ["//visibility:public"])

py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        "//vaeseq:eval_core",
        "//vaeseq:gen_core",
        "//vaeseq:util",
    ],
)

py_library(
    name = "independent_sequence",
    srcs = ["independent_sequence.py"],
    deps = [
        ":base",
        "//vaeseq:dist_module",
        "//vaeseq:latent",
        "//vaeseq:util",
    ],
)

py_library(
    name = "rnn",
    srcs = ["rnn.py"],
    deps = [
        ":base",
        "//vaeseq:dist_module",
        "//vaeseq:util",
    ],
)

py_library(
    name = "srnn",
    srcs = ["srnn.py"],
    deps = [
        ":base",
        "//vaeseq:dist_module",
        "//vaeseq:latent",
        "//vaeseq:util",
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
        "//vaeseq:agent",
        "//vaeseq:codec",
        "//vaeseq:hparams",
        "//vaeseq:util",
    ],
)
