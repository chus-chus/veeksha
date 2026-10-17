# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add project root to sys.path so we can import veeksha
sys.path.insert(0, str(Path(__file__).parents[1].resolve()))

from veeksha.cli.config_docs_generator import generate_sphinx_docs

def setup(app):
    """Run config docs generator before build."""
    def run_config_gen(app):
        print("[veeksha] Generating config documentation...")
        generate_sphinx_docs(output_dir=str(Path(app.confdir) / "config_reference"))

    app.connect("builder-inited", run_config_gen)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "veeksha"
copyright = "2024-onwards Systems for AI Lab, Georgia Institute of Technology"
author = "Veeksha Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Added configurations ----------------------------------------------------
html_title = "veeksha"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "chus-chus",  # GitHub org/user
    "github_repo": "veeksha",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}
