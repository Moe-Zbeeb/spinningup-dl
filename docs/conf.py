project = "Spinning Up in Deep Learning"
author = "AUB Frontier AI Fellows"
copyright = "2026, AUB Frontier AI Fellows"

github_user = "Moe-Zbeeb"
github_repo = "spinningup-dl"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "amsmath",
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

mathjax3_config = {
    "tex": {
        "macros": {
            "R": r"\mathbb{R}",
            "E": r"\mathop{\mathbb{E}}",
            "L": r"\mathcal{L}",
            "argmax": r"\mathop{\mathrm{arg\,max}}",
            "argmin": r"\mathop{\mathrm{arg\,min}}",
        }
    }
}

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 3,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
}
html_title = project
html_static_path = ["_static"]
html_css_files = ["css/faf.css"]
html_favicon = "_static/img/aub-faf-icon.png"
html_baseurl = f"https://{github_user.lower()}.github.io/{github_repo}/"
html_show_sourcelink = False
html_copy_source = False
html_context = {
    "display_github": True,
    "github_user": github_user,
    "github_repo": github_repo,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

pygments_style = "default"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
