# Configuration file for the Sphinx documentation builder.
import errno
import os
import shutil
import subprocess
import sys

from natsort import natsorted
from recommonmark.parser import CommonMarkParser

docs_dir = os.path.dirname(__file__)
repodir = os.path.abspath(os.path.join(__file__, r"../../.."))
gitdir = os.path.join(repodir, r".git")

# -- Project information -----------------------------------------------------

project = "Merlin"
copyright = "2022, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_multiversion",
    "sphinx_rtd_theme",
    "recommonmark",
    "sphinx_markdown_tables",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False

# Whitelist pattern for tags (set to None to ignore all tags)
# Determine if Sphinx is reading conf.py from the checked out
# repo (a Git repo) vs SMV reading conf.py from an archive of the repo
# at a commit (not a Git repo).
if os.path.exists(gitdir):
    tag_refs = (
        subprocess.check_output(["git", "tag", "-l", "v*"]).decode("utf-8").split()
    )
    tag_refs = natsorted(tag_refs)[-6:]
    smv_tag_whitelist = r"^(" + r"|".join(tag_refs) + r")$"
else:
    # SMV is reading conf.py from a Git archive of the repo at a specific commit.
    smv_tag_whitelist = r"^v.*$"

# Only include main branch for now
smv_branch_whitelist = "^main$"

html_sidebars = {"**": ["versions.html"]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

source_parsers = {".md": CommonMarkParser}
source_suffix = [".rst", ".md"]

nbsphinx_allow_errors = True
html_show_sourcelink = False

autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
    "member-order": "bysource",
}

autosummary_generate = True


def copy_files(src: str):
    """
    src_dir: A path, specified as relative to the
             docs/source directory in the repository.
             The source can be a directory or a file.
             Sphinx considers all directories as relative
             to the docs/source directory.

             TIP: Add these paths to the .gitignore file.
    """
    src_path = os.path.abspath(src)
    if not os.path.exists(src_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), src_path)
    out_path = os.path.basename(src_path)
    out_path = os.path.abspath("{}/".format(out_path))

    print(
        r"Copying source documentation from: {}".format(src_path),
        file=sys.stderr,
    )
    print(r"  ...to destination: {}".format(out_path), file=sys.stderr)

    if os.path.exists(out_path) and os.path.isdir(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    if os.path.exists(out_path) and os.path.isfile(out_path):
        os.unlink(out_path)

    if os.path.isdir(src_path):
        shutil.copytree(src_path, out_path)
    else:
        shutil.copyfile(src_path, out_path)


copy_files(r"../../README.md")
copy_files(r"../../examples/")
