from setuptools import setup, find_packages

setup(
    name="NeuroInsight",
    author="Pranjal Khadka",
    author_email="pranzalkhadka1@gmail.com",
    version="0.1",
    description="A custom neural network library for educational purposes",
    long_description="NeuroInsight is a simple neural network library designed to help beginners understand the internals of neural networks.",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'plotly'
    ],
)
