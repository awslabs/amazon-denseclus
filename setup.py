import os
import setuptools
from collections import OrderedDict

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()


# Read __init__.py
version_py = os.path.join("denseclus", "__version__.py")
with open(version_py) as fp:
    lines = fp.readlines()
meta = OrderedDict()
for line in lines:
    if "pylint" in line:
        continue
    key, value = line.split("=")
    meta[key.strip()] = value.strip()[1:-1]


setuptools.setup(
    name=meta["__title__"],
    version=meta["__version__"],
    author=meta["__author__"],
    description=meta["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url=meta["__url__"],
    packages=setuptools.find_packages(include=["denseclus", "denseclus.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10, <3.12",
    license_files=meta["__license__"],
    install_requires=required,
    extras_require={
        "test": ["pytest", "pytest-cov"],
        "gpu-cu12": ["cuml-cu12"],
        "gpu-cu11": ["cuml-cu11"],
    },
    keywords="amazon dense clustering",
    project_urls={
        "Bug Tracker": "https://github.com/awslabs/amazon-denseclus/issues",
        "Documentation": "https://github.com/awslabs/amazon-denseclus/notebooks",
        "Source Code": "https://github.com/awslabs/amazon-denseclus",
    },
    platforms=["any"],
)
