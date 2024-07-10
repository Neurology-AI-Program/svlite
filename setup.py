from setuptools import setup, find_packages
from svlite._version import __version__

setup(
    name = 'svlite.py',
    packages = find_packages(),
    author = 'The Mayo Clinic Neurology AI Program',
    version = __version__
)