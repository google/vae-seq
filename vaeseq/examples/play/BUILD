test_suite(
    name = "tests",
    tests = [
        "environment_test",
        "model_test",
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
    name = "agent",
    srcs = ["agent.py"],
    deps = [
        "//vaeseq:agent",
        "//vaeseq:util",
    ],
)

py_library(
    name = "environment",
    srcs = ["environment.py"],
    deps = [
        "//vaeseq:agent",
        "//vaeseq:util",
    ],
)

py_test(
    name = "environment_test",
    srcs = ["environment_test.py"],
    deps = [
        ":environment",
        "//vaeseq:util",
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
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":agent",
        ":codec",
        ":environment",
        "//vaeseq:model",
        "//vaeseq:util",
        "//vaeseq/vae",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":hparams",
        ":model",
        "//vaeseq:model_test",
    ],
)

py_binary(
    name = "play",
    srcs = ["play.py"],
    deps = [
        ":hparams",
        ":model",
    ],
)
