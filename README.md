# Spinning Up in Deep Learning

An educational resource for learning deep learning, from [AUB Frontier AI Fellows](https://aub-faf.com/) at the American University of Beirut. Its structure follows OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com/).

This repository holds the website. Pages are written in Markdown under `docs/` and built with Sphinx.

## Run it locally

```bash
make setup
make serve
```

`make setup` creates `.venv` and installs the build tools (Python 3.12 or newer). `make serve` opens the site at `http://localhost:8000` and reloads when a file changes.

| Command | What it does |
| --- | --- |
| `make setup` | Create the virtual environment and install dependencies |
| `make serve` | Live preview with auto-reload |
| `make html` | Build once into `docs/_build/html` |
| `make check` | Build with warnings as errors, as the deploy does |
| `make clean` | Delete the build output |

## Add content

Each page is one Markdown file. Most pages currently hold a "Coming soon" box: open the file, replace the box with the content, and push.

- `docs/index.md` controls the landing page and the sidebar order.
- `docs/etc/style-guide.md` shows every element a page can use (math, callout boxes, code, tables, figures, footnotes). Copy from it.
- `.claude/skills/add-content/SKILL.md` is the full guide: how to add, rename, and remove pages, the writing conventions, and the fonts and colors. Claude Code loads it automatically when asked to add content in this repository.

## Layout

```
docs/
  conf.py               site configuration
  index.md              landing page and sidebar
  user/                 User Documentation
  intro/                Introduction to Deep Learning
  resources/            Resources
  models/               Models Docs
  utils/                Utilities Docs
  etc/                  Acknowledgements, About, element reference
  _static/css/modify.css  all styling, ported from Spinning Up
  _static/img/          images
  _templates/           breadcrumbs and footer line
tools/                  sources for the logo image and the banner
.github/workflows/      build and deploy to GitHub Pages
.claude/skills/         the add-content guide
```

## Deploy

Every push to `main` runs `.github/workflows/deploy.yml`, which builds the site and publishes it to GitHub Pages. In the repository settings, Pages must have its source set to "GitHub Actions".

## Credits

Named after and modelled on [Spinning Up in Deep RL](https://spinningup.openai.com/) by Joshua Achiam (OpenAI, MIT License). The stylesheet is a port of Spinning Up's; the name, logo, and banner are AUB Frontier AI Fellows.
