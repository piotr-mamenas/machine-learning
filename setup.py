# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Machine Learning',
    version='0.0.1',
    description='The machine learning system behind Bubbles',
    long_description=readme,
    author='Piotr Mamenas',
    author_email='piotr.mamenas@gmail.com',
    url='https://github.com/piotr-mamenas/machine-learning',
    license=license,
    packages=find_packages(exclude=('tests'))
)