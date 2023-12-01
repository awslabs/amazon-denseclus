# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Amazon DenseClus"
copyright = "2023, Charles Frenzel and Baichuan Sun"
author = "Charles Frenzel and Baichuan Sun"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.mathjax",
    # other extensions...
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"  # Change the theme
html_logo = "logo.png"  # Add a logo image file in _static directory

html_theme_options = {  # Customize the sidebar
    "description": "Amazon DenseClus Documentation",
    "github_user": "awslabs",
    "github_repo": "amazon-denseclus",
}

html_static_path = ["_static"]

autodoc_default_options = {
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
