# Configuration file for the Sphinx documentation builder.
import os
import re
import subprocess
import sys

from natsort import natsorted

sys.path.insert(0, os.path.abspath("/models/"))
docs_dir = os.path.dirname(__file__)
repodir = os.path.abspath(os.path.join(__file__, r"../../.."))
gitdir = os.path.join(repodir, r".git")

# -- Project information -----------------------------------------------------

project = "Merlin"
copyright = "2023, NVIDIA"  # pylint: disable=redefined-builtin
author = "NVIDIA"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_multiversion",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_external_toc",
    "sphinxcontrib.copydirs",
]

# MyST configuration settings
external_toc_path = "toc.yaml"
myst_enable_extensions = [
    "deflist",
    "html_image",
    "linkify",
    "replacements",
    "tasklist",
]
myst_linkify_fuzzy_links = False
myst_heading_anchors = 3
nb_execution_mode = "off"

# Some documents are RST and include `.. toctree::` directives.
suppress_warnings = ["etoc.toctree", "myst.header", "misc.highlighting_failure"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["examples/legacy"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = "NVIDIA Merlin"
html_favicon = "_static/favicon.png"
html_theme_options = {
    "repository_url": "https://github.com/NVIDIA-Merlin/Merlin",
    "use_repository_button": True,
    "footer_content_items": ["copyright.html", "last-updated.html"],
    "extra_footer": "",
    "logo": {"text": "NVIDIA Merlin", "alt_text": "NVIDIA Merlin"},
}
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "search-field.html",
        "icon-links.html",
        "sbt-sidebar-nav.html",
        "merlin-ecosystem.html",
        "versions.html",
    ]
}
html_css_files = ["css/custom.css", "css/versions.css"]
html_js_files = ["js/rtd-version-switcher.js"]
html_context = {"analytics_id": "G-NVJ1Y1YJHK"}
html_copy_source = False
html_show_sourcelink = False

# Whitelist pattern for tags (set to None to ignore all tags)
# Determine if Sphinx is reading conf.py from the checked out
# repo (a Git repo) vs SMV reading conf.py from an archive of the repo
# at a commit (not a Git repo).
if os.path.exists(gitdir):
    tag_refs = subprocess.check_output(["git", "tag", "-l", "v*"]).decode("utf-8").split()
    tag_refs = [tag for tag in tag_refs if re.match(r"^v[0-9]+.[0-9]+.[0-9]+$", tag)]
    tag_refs = natsorted(tag_refs)[-6:]
    smv_tag_whitelist = r"^(" + r"|".join(tag_refs) + r")$"
else:
    # SMV is reading conf.py from a Git archive of the repo at a
    # specific commit.
    smv_tag_whitelist = r"^v.*$"

smv_branch_whitelist = "^(main|stable)$"

smv_refs_override_suffix = "-docs"

html_baseurl = "https://nvidia-merlin.github.io/Merlin/stable/"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

source_suffix = [".rst", ".md"]

nbsphinx_allow_errors = True

autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
    "member-order": "bysource",
}

autosummary_generate = True

copydirs_additional_dirs = ["../../examples/", "../../README.md"]

copydirs_file_rename = {
    "README.md": "index.md",
}

# Generate the support matrix tables.
proc = subprocess.run(["python", "docs/smx2rst.py"], cwd=repodir, check=True)
if proc.returncode != 0:
    print("Failed to generate support matrix table snippets.", file=sys.stderr)
    sys.exit(1)
