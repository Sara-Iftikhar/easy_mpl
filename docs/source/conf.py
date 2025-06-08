# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from sphinx_gallery.sorting import ExampleTitleSortKey, ExplicitOrder

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'easy_mpl'
copyright = '2022, Ather Abbas'
author = 'Ather Abbas, Sara Iftikhar'

# The full version, including alpha/beta/rc tags
release = "0.21.5"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.intersphinx',
  'sphinx.ext.ifconfig',
  'sphinx.ext.viewcode',
  'sphinx_toggleprompt',
  'sphinx_copybutton',
  "sphinx-prompt",
  'sphinx.ext.napoleon',
'sphinx_gallery.gen_gallery',
]

napoleon_numpy_docstring = True
toggleprompt_offset_right  = 30

# specify the master doc, otherwise the build at read the docs fails
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

sphinx_gallery_conf = {
    'backreferences_dir': 'gen_modules/backreferences',
    #'doc_module': ('sphinx_gallery', 'numpy'),
    'reference_url': {
        'sphinx_gallery': None,
    },
    'examples_dirs': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples'),
    'gallery_dirs': 'auto_examples',
    'compress_images': ('images', 'thumbnails'),
    'filename_pattern': '',

    'binder': {'org': 'sphinx-gallery',
               'repo': 'sphinx-gallery.github.io',
               'branch': 'master',
               'binderhub_url': 'https://mybinder.org',
               'dependencies': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.binder', 'requirements.txt'),
               'notebooks_dir': 'notebooks',
               'use_jupyter_lab': True,
               },
    'show_memory': True,
    # capture raw HTML or, if not present, __repr__ of last expression in
    # each code block
    'capture_repr': ('_repr_html_', '__repr__'),
    'matplotlib_animations': True,
    'image_srcset': ["2x"],

    'within_subsection_order':  ExampleTitleSortKey,
        # '../../examples/plot',
        # '../../examples/scatter',
        # '../../examples/imshow',
        # '../../examples/bar_chart',
        # '../../examples/hist',
        # '../../examples/lollipop_plot',
        # '../../examples/dumbell_plot',
        # '../../examples/regplot',
        # '../../examples/ridge',
        # '../../examples/pie',
        # '../../examples/contour',
        # '../../examples/circular_bar_plot',
        # '../../examples/parallel_coordinates',
        # '../../examples/spider_plot',
        # '../../examples/taylor_plot',
#                                       ]),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']