# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('./API/'))

project = 'ADFWI Document'
copyright = '2025, Liu Feng'
author = 'Liu Feng'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
              #'recommonmark',
              'myst_parser',
              'sphinx.ext.mathjax',
              'sphinx_markdown_tables',
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc', 
              'sphinx.ext.viewcode']

templates_path = ['_templates']
exclude_patterns = []

# for Sphinx-1.3
from recommonmark.parser import CommonMarkParser

source_parsers = {
    '.md': CommonMarkParser,
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    #'.md' : 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# for using Read the Docs theme
# html_theme = 'sphinxdoc'
#html_theme_path = []

# import sphinx_rtd_theme
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'