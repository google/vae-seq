# Everything depends on TensorFlow and Sonnet.
package(default_visibility = ["//visibility:public"])

test_suite(
    name = "tests",
    tests = [
          "agent_test",
          "util_test",
          "//vae_seq/vae:vae_test",
    ],
)

py_library(
    name = "hparams",
    srcs = ["hparams.py"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
         ":hparams",
         ":util",
    ],
)

py_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
         ":util",
    ],
)

py_test(
    name = "agent_test",
    srcs = ["agent_test.py"],
    deps = [
        ":agent",
        ":codec",
        ":hparams",
    ],
)

py_library(
    name = "codec",
    srcs = ["codec.py"],
    deps = [
        ":util",
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":util",
    ],
)
