from setuptools import setup, find_namespace_packages
import re

# read the contents of your README file
import os
from os import path

with open("README.md") as f:
    long_description = f.read()

verstr = "unknown"
try:
    verstrline = open("causal_covid/_version.py", "rt").read()
except EnvironmentError:
    pass
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("unable to find version in causal_covid/_version.py")


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name="causal_covid",
    author="Jonas Dehning, Simon Bauer, Viola Priesemann",
    author_email="jonas.dehning@ds.mpg.de",
    packages=find_namespace_packages(),
    url="https://github.com/Priesemann-Group/causal_covid",
    description="Toolbox to explore vaccination scenarios",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6.0",
    version=verstr,
    install_requires=parse_requirements("./requirements.txt"),
)
