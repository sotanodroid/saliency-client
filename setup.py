#!/usr/bin/env python

import os

from setuptools import setup, find_packages

__version__ = "0.1"

setup(name='saliency-client',
      version=__version__,
      description='Client for connecting to Saliency API',
      author='Saliency.ai',
      author_email='lukasz@saliency.ai',
      url='http://saliency.ai/',
      license='Apache 2.0',
      packages=find_packages(),
      install_requires=['numpy>=1.14.2','pandas>=0.25.1'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          ],
)
