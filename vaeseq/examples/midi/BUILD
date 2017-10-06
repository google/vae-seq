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
        "//vaeseq:batch_dist",
        "//vaeseq:codec",
        "//vaeseq:dist_module",
        "//vaeseq:util",
    ],
)

py_library(
    name = "hparams",
    srcs = ["hparams.py"],
    deps = [
        "//vaeseq:hparams",
    ]
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":codec",
        "//vaeseq/vae",
        "//vaeseq:agent",
        "//vaeseq:train",
        "//vaeseq:util",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":dataset",
        ":hparams",
        ":model",
        "//vaeseq:util",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":dataset",
        ":hparams",
        ":model",
        "//vaeseq:util",
    ],
)

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    deps = [
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
        "//vaeseq:util",
    ],
)
