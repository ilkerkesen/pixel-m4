#!/usr/bin/env python

# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="pixel",
    version="0.1.0",
    author="PIXEL-M4 Authors",
    author_email="ilke@di.ku.dk",
    url="https://github.com/ilkerkesen/pixel-m4",
    description="Research code for the paper 'Multilingual Pretraining for Pixel Language Models'",
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages("pixel"),
    zip_safe=True,
)
