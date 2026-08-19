"""Sphinx configuration for the CompositeOT documentation."""

from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

project = "CompositeOT"
author = "CompositeOT contributors"
copyright = "2026, CompositeOT contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "generated/CompositeOT.hyperparams.rst",
    "generated/CompositeOT.linesearch.rst",
    "generated/CompositeOT.marginal.rst",
    "generated/CompositeOT.newton.rst",
    "generated/CompositeOT.normal.rst",
    "generated/CompositeOT.normalassemble.rst",
    "generated/CompositeOT.normallowrank.rst",
    "generated/CompositeOT.ops.rst",
    "generated/CompositeOT.palm.rst",
    "generated/CompositeOT.plan.rst",
    "generated/CompositeOT.scaling.rst",
    "generated/CompositeOT.sgsadmm.rst",
    "generated/CompositeOT.side.rst",
    "generated/CompositeOT.subproblem.rst",
    "generated/CompositeOT.termination.rst",
    "generated/CompositeOT.verbose.rst",
]

html_theme = "alabaster"
html_title = "CompositeOT"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "description": "A Python solver for composite optimal transport.",
    "fixed_sidebar": True,
    "show_powered_by": False,
    "page_width": "1080px",
    "sidebar_width": "260px",
}
