#!/usr/bin/env/python3
import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Amazon DenseClus",
    version="0.0.19",
    author="Charles Frenzel",
    description="Dense Clustering for Mixed Data Types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awslabs/amazon-denseclus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    license_files=("LICENSE",),
    install_requires=[
        "umap_learn>=0.5.1",
        "numpy>=1.20.2",
        "hdbscan>=0.8.27",
        "numba>=0.51.2",
        "pandas>=1.2.4",
        "scikit_learn>=0.24.2",
    ],
)
