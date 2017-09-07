# Copyright 2018 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    ],
    extras_require={
        "tf": ["tensorflow>=1.4.0"],
        "tf_gpu": ["tensorflow-gpu>=1.4.0"],
    },
    entry_points={
        "console_scripts": [
            "vaeseq-text = vaeseq.examples.text.text:main",
            "vaeseq-midi = vaeseq.examples.midi.midi:main",
            "vaeseq-play = vaeseq.examples.play.play:main",
        ],
    },
    test_suite="setup.tests",
)
