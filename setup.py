from setuptools import setup, find_packages

setup(
    name = "NeuroInsight",
    author="Pranjal Khadka",
    author_email="pranzalkhadka1@gmail.com",
    version = "0.1",
    packages = find_packages(),
    install_requires = [
        'numpy',
        'matplotlib'
    ]
)