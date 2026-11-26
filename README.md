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
| `make lint` | Check that every page follows the template |
| `make check` | Run the template check, then build with warnings as errors, as the deploy does |
| `make shot PAGE=intro/neural_networks` | Save a screenshot of a built page to `docs/_build/shots/` |
| `make clean` | Delete the build output |

## Companion code

The `spinup_dl` package has standalone PyTorch implementations of the models in the Models Docs section: an MLP, ConvNets (LeNet and a ResNet), character-level RNN/GRU/LSTM and Transformer language models, a VAE, and a diffusion model.

```bash
pip install -e .
python -m spinup_dl.run mlp --epochs 3
python -m spinup_dl.utils.plot data/mlp --value TestAccuracy
```

The site's Installation, Models, Running Experiments, Experiment Outputs, and Plotting pages document it. `pytest` runs the tests, which `.github/workflows/tests.yml` also runs on every change to the code.

## Add content

Each page is one Markdown file. Most pages currently hold a "Coming soon" box: open the file, replace the box with the content, and push.

- `docs/index.md` controls the landing page and the sidebar order.
- `docs/etc/style-guide.md` shows every element a page can use (math, callout boxes, code, tables, figures, footnotes). Copy from it.
- `templates/` holds the shape every page and section follows: `part.md`, `section.md`, `model.md`, and `placeholder.md`.
- `AGENTS.md` is the brief for AI coding agents (Claude Code, Codex, Cursor and others): how to turn notes into a section, and the house style that keeps the site reading like Spinning Up. `CLAUDE.md` points Claude Code at it.
- `.claude/skills/add-content/SKILL.md` has the mechanics: how to add, rename, and remove pages, the syntax of each element, and the fonts and colors.
- `tools/lint_content.py` enforces the template. It runs in `make check` and in the deploy, so an off-template page cannot be published.

## Layout

```
docs/
  conf.py               site configuration
  index.md              landing page and sidebar
  user/                 User Documentation
  intro/                Introduction to Deep Learning
  resources/            Resources
  models/               Models Docs
  etc/                  Acknowledgements, About, element reference
  _static/css/modify.css  all styling, ported from Spinning Up
  _static/img/          images
  _templates/           breadcrumbs and footer line
spinup_dl/              companion code: models, runner, logger, plotting
tests/                  tests for the companion code
templates/              page and section templates
tools/                  template checker, and sources for the logo and banner
AGENTS.md               instructions for AI coding agents
.github/workflows/      build and deploy to GitHub Pages
.claude/skills/         the add-content guide
```

## Deploy

Every push to `main` runs `.github/workflows/deploy.yml`, which builds the site and publishes it to GitHub Pages. In the repository settings, Pages must have its source set to "GitHub Actions".

## Credits

Named after and modelled on [Spinning Up in Deep RL](https://spinningup.openai.com/) by Joshua Achiam (OpenAI, MIT License). The course follows the topics of Andrew Ng's Deep Learning Specialization; see the Acknowledgements page for full credits. The stylesheet is a port of Spinning Up's; the name and logo are AUB Frontier AI Fellows.
