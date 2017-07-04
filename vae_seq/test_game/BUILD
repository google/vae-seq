py_library(
    name = "game",
    srcs = ["game.py"],
    deps = [
        # NumPy
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":game",
        "//vae_seq:obs_layers",
        "//vae_seq:model",
    ],
)

py_binary(
    name = "test_game",
    srcs = ["test_game.py"],
    deps = [
        ":model",
        "//vae_seq:hparams",
    ],
)
