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
        "dm-sonnet>=1.10",
        "future>=0.16.0",
        "gym>=0.9.3",
        "numpy>=1.12.0",
        "pretty-midi>=0.2.8",
        "scipy>=0.16.0",
        "six>=1.0.0",
        "tensorflow>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "vaeseq-text = vaeseq.examples.text.text:main",
            "vaeseq-midi = vaeseq.examples.midi.midi:main",
            "vaeseq-play = vaeseq.examples.play.play:main",
        ],
    },
    test_suite="setup.tests",
)
