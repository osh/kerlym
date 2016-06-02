#!/usr/bin/env python
import os
from setuptools import setup
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "KeRLym",
    version = "0.0.1",
    author = "Tim O'Shea",
    author_email = "tim.oshea753@gmail.com",
    description = ("Keras Reinforcement Learners for Gym."),
    license = "MIT",
    keywords = "keras reinforcement learning gym",
    url = "http://www.kerlym.com",
    packages=['kerlym'],
    long_description=read('README.md'),
    scripts=['kerlym/kerlym'],
)
