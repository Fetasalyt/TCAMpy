# Configuration file for the Sphinx documentation builder.

# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = 'TCAMpy'
copyright = '2025, Rózsa Richárd Bence'
author = 'Rózsa Richárd Bence'


# -- General configuration ---------------------------------------------------

extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx_rtd_theme',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
