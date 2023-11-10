#!/usr/bin/env/python3
import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="Amazon DenseClus",
    version="0.1.0",
    author="Charles Frenzel & Baichuan Sun",
    description="Dense Clustering for Mixed Data Types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awslabs/amazon-denseclus",
    packages=setuptools.find_packages(include=["denseclus", "denseclus.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Data Scientists",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.10, <3.12",
    license_files=("LICENSE",),
    install_requires=required,
    keywords="amazon dense clustering",
    project_urls={
        "Bug Tracker": "https://github.com/awslabs/amazon-denseclus/issues",
        "Documentation": "https://github.com/awslabs/amazon-denseclus/notebooks",
        "Source Code": "https://github.com/awslabs/amazon-denseclus",
    },
    platforms=["any"],
)
