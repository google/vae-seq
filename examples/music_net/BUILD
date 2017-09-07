test_suite(
    name = "tests",
    tests = [
        "agent_test",
        "dataset_test",
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
    name = "agent",
    srcs = ["agent.py"],
    deps = [
         "//vae_seq:agent",
         "//vae_seq:util",
    ],
)

py_test(
    name = "agent_test",
    srcs = ["agent_test.py"],
    deps = [
         ":agent",
         ":codec",
         ":hparams",
         "//vae_seq:util",
    ]
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
        ":agent",
        ":codec",
        "//vae_seq/vae",
        "//vae_seq:hparams",
        "//vae_seq:train",
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
