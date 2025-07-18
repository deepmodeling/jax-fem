# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..')) 

if '/docs' in str(sys.path):
    sys.path.remove('/docs')


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'JAX-FEM'
copyright = '2025, JAX-FEM team'
author = 'JAX-FEM team'
release = '0.0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode', 
              'sphinx.ext.napoleon',
              "sphinx.ext.mathjax",
              "sphinx_copybutton",
              "myst_parser",
              'nbsphinx']

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'modules': ['jax_fem'],
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'imported-members': False,
    'special-members': False,
}

autodoc_mock_imports = [
    "jax",
    "numpy",
    "basix",
    "scipy",
    "matplotlib",
    "meshio",
    "petsc4py",
    "fenics",
    "gmsh",
    "fenics_basix",
    "pyfiglet"
]

autodoc_typehints = 'description'

autodoc_member_order = 'bysource'

# autodoc
autodoc_docstring_signature = False

autosummary_generate = True

# ipynb
nbsphinx_execute = 'never'

html_sourcelink_suffix = ''

nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. raw:: html

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const downloadLinks = document.querySelectorAll('a.reference.download.internal');
        downloadLinks.forEach(link => {
            link.setAttribute('download', '');
            link.href = '../_sources/' + link.getAttribute('href');
        });
    });
    </script>
"""


highlight_language = 'python'

python_use_unqualified_type_names = True

myst_enable_extensions = [
    "amsmath",             
    "dollarmath",          
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/deepmodeling/jax-fem",  
    "use_repository_button": True,     # GitHub
    # "use_edit_page_button": True,      # "Edit on GitHub" 
    # "use_issues_button": True,         # "Open Issue"            
    "show_navbar_depth": 1,  
    "navigation_depth": 5,   
    "collapse_navigation": True,  
    "globaltoc_includehidden": True,                  
}


html_static_path = ['_static'] 

html_css_files = [
    'custom.css',  
]