# Everything depends on TensorFlow and Sonnet.
package(default_visibility = ["//visibility:public"])

test_suite(
    name = "tests",
    tests = [
          "agent_test",
          "batch_dist_test",
          "util_test",
          "//vaeseq/examples/midi:tests",
          "//vaeseq/examples/text:tests",
          "//vaeseq/vae:vae_test",
    ],
)

py_library(
    name = "dist_module",
    srcs = ["dist_module.py"],
)

py_library(
    name = "batch_dist",
    srcs = ["batch_dist.py"],
)

py_test(
    name = "batch_dist_test",
    srcs = ["batch_dist_test.py"],
    deps = [
         ":batch_dist",
    ],
)

py_library(
    name = "latent",
    srcs = ["latent.py"],
    deps = [
         ":dist_module",
         ":util",
    ],
)

py_library(
    name = "gen_core",
    srcs = ["gen_core.py"],
    deps = [
         ":agent",
         ":dist_module",
         ":util",
    ],
)

py_library(
    name = "eval_core",
    srcs = ["eval_core.py"],
    deps = [
         ":agent",
         ":dist_module",
         ":util",
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
        ":dist_module",
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
