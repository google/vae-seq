from setuptools import setup, find_packages
import unittest


def tests():
    """Used by test_suite below."""
    return unittest.TestLoader().discover(
        "vaeseq/", "*_test.py", top_level_dir=".")


setup(
    name="vae-seq",
    author="Yury Sulsky",
    author_email="yury.sulsky@gmail.com",
    version="0.1",
    description="Generative Sequence Models",
    long_description=open("README.md").read(),
    packages=find_packages(),
    install_requires=[
        "future>=0.16.0",
        "numpy>=1.12.0",
        "tensorflow>=1.3.0",
        "dm-sonnet>=1.10",
        "pretty-midi>=0.2.8",
        "scipy>=0.16.0",
    ],
    entry_points={
        "console_scripts": [
            # examples/text
            "vaeseq-text-train = vaeseq.examples.text.train:entry_point",
            "vaeseq-text-generate = vaeseq.examples.text.generate:entry_point",
            "vaeseq-text-eval = vaeseq.examples.text.evaluate:entry_point",

            # examples/midi
            "vaeseq-midi-train = vaeseq.examples.midi.train:entry_point",
            "vaeseq-midi-generate = vaeseq.examples.midi.generate:entry_point",
            "vaeseq-midi-eval = vaeseq.examples.midi.evaluate:entry_point",
        ],
    },
    test_suite="setup.tests",
)
