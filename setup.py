from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="rait",
    version="0.9.0",

    description="Python implementation of RAIT from matlab",
    url="https://github.com/Nguyen-Thac-Bach/RAIT-Python",
    
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'torch>=2.3',
    ],
    author="Thac Bach Nguyen",
)