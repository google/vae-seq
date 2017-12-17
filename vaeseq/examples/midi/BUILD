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
    name = "hparams",
    srcs = ["hparams.py"],
    deps = [
        "//vaeseq:hparams",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":dataset",
        "//vaeseq:codec",
        "//vaeseq:context",
        "//vaeseq:model",
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
        "//vaeseq:model_test",
        "//vaeseq:util",
    ],
)

py_binary(
    name = "midi",
    srcs = ["midi.py"],
    deps = [
        ":hparams",
        ":model",
    ],
)
