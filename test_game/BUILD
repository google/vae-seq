test_suite(
    name = "tests",
    tests = [
        "model_test",
    ],
)

py_library(
    name = "game",
    srcs = ["game.py"],
    deps = [
        # NumPy
    ],
)

py_library(
    name = "environment",
    srcs = ["environment.py"],
    deps = [
        ":game",
    ],
)

py_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
        ":game",
        "//vae_seq:agent",
    ],
)

py_library(
    name = "hparams",
    srcs = ["hparams.py"],
    deps = [
        "//vae_seq:hparams",
    ])

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":agent",
        ":environment",
        ":game",
        "//vae_seq/vae",
        "//vae_seq:obs_layers",
        "//vae_seq:train",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":hparams",
        ":model",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":hparams",
        ":model",
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
