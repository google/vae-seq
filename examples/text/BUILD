test_suite(
    name = "tests",
    tests = [
        "dataset_test",
        "model_test",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    deps = [
         ":dataset",
    ],
)

py_library(
    name = "codec",
    srcs = ["codec.py"],
    deps = [
        "//vae_seq:dist_module",
        "//vae_seq:util",
    ],
)

py_library(
    name = "hparams",
    srcs = ["hparams.py"],
    deps = [
        "//vae_seq:hparams",
    ]
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":codec",
        "//vae_seq/vae",
        "//vae_seq:agent",
        "//vae_seq:train",
        "//vae_seq:util",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":dataset",
        ":hparams",
        ":model",
        "//vae_seq:util",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":dataset",
        ":hparams",
        ":model",
        "//vae_seq:util",
    ],
)

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    deps = [
        ":dataset",
        ":hparams",
        ":model",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        ":dataset",
        ":hparams",
        ":model",
        "//vae_seq:util",
    ],
)
