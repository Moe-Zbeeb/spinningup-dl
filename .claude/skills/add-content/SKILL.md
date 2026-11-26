---
name: add-content
description: Use when adding, writing, editing, renaming, moving, or removing a page, section, figure, equation, code block, or callout box on the Spinning Up in Deep Learning site (the Sphinx and MyST Markdown site under docs/), when filling in a "Coming soon" placeholder page, or when asked what font, colors, or page conventions the site uses.
---

# Adding Content to Spinning Up in Deep Learning

## Overview

The site is a Sphinx project in `docs/`, written in MyST Markdown, themed with `sphinx_rtd_theme` and one stylesheet that applies the AUB Frontier AI Fellows (AUB–FAF) look. A page is one `.md` file. A page appears in the sidebar only when a `toctree` in `docs/index.md` lists it.

`docs/etc/style-guide.md` is the live reference for every element (math, callouts, code, tables, figures, footnotes). Read its source before writing a page and copy from it. It renders at `/etc/style-guide.html`.

## Where Things Live

| What | Where |
| --- | --- |
| Landing page and sidebar order | `docs/index.md` |
| Pages | One folder per sidebar section: `docs/user/`, `docs/intro/`, `docs/resources/`, `docs/models/`, `docs/utils/`, `docs/etc/`. New sections get a new folder |
| Element reference page | `docs/etc/style-guide.md` |
| Images | `docs/_static/img/<page-name>/` |
| All styling | `docs/_static/css/faf.css` |
| Sidebar brand block, footer line | `docs/_templates/layout.html`, `docs/_templates/footer.html` |
| Site name, extensions, math macros, GitHub repo | `docs/conf.py` |
| Deploy on push to `main` | `.github/workflows/deploy.yml` |

## Tasks

### Fill in a placeholder page

Every unwritten page holds a single `Coming soon` box. Open the file, delete the whole ` ```{admonition} Coming soon ` block, keep the `# Title` line, and write the page below it. Nothing else needs to change: the sidebar and landing page read the title and the `##` headings from the file.

### Add a new page to an existing section

1. Create `docs/<section>/<name>.md`. Use a short lowercase file name with underscores, such as `docs/models/resnet.md`. The file name becomes the URL.
2. Start the file with one `# Title`. That title is the sidebar label.
3. If the page is not written yet, give it the placeholder box below the title and nothing else:

   ````markdown
   # Residual Network

   ```{admonition} Coming soon
   :class: coming-soon

   This page is being written. It will cover background, key equations, pseudocode, and documentation for residual networks.
   ```
   ````

4. Add `<section>/<name>` (no extension) to the matching `toctree` in `docs/index.md`, at the position it should appear in the sidebar.

### Add a new section to the sidebar

1. Create a folder `docs/<section>/` and at least one page file in it, as in the steps above.
2. Add a `toctree` block to `docs/index.md`. The `:caption:` is the section label. Sections appear in the sidebar in the order their blocks appear in the file, so place the block where the section belongs. Keep `:maxdepth: 2`. Every line in the block must match a file that exists.

````markdown
```{toctree}
:maxdepth: 2
:caption: Advanced Topics

advanced/normalization
```
````

### Rename, move, or remove a page

Rename or delete the file, update its line in `docs/index.md`, then search `docs/` for links to the old path and fix them. `make check` fails on any link or toctree entry that points to a missing page.

### Publish a page that is not ready

Leave the `Coming soon` box in place, or keep the page out of every `toctree` and put this at the very top of the file so the build does not warn about it:

```markdown
---
orphan: true
---
```

## Page Shape

One `#` title per page, then `##` sections, then `###` subsections. The sidebar and the landing page show `##` headings under each page, so write them as short labels in Title Case, as Spinning Up does ("Key Concepts and Terminology", not "Let's look at some key concepts").

A concept page (the Introduction to Deep Learning parts) reads top to bottom: what the reader will learn, the ideas in the order they depend on each other, a recap.

A Models Docs page follows the Spinning Up algorithm-page outline:

```markdown
# Multilayer Perceptron

## Background

### Quick Facts

### Key Equations

### Pseudocode

## Documentation

## References

### Relevant Papers

### Other Public Implementations
```

| Heading | What goes under it |
| --- | --- |
| Background | What the model is, what problem it was built for, and what the reader should already know, with a link to the page that teaches it |
| Quick Facts | Three to five bullets: input and output, what the model assumes about the data, when to use it |
| Key Equations | The forward computation and the loss, each symbol defined right after the equation |
| Pseudocode | The algorithm in a `text` block |
| Documentation | A minimal PyTorch implementation with a usage example. When the site gains a companion code package, this becomes the reference for that package's implementation |
| Relevant Papers | A plain bulleted list of the papers that introduced or shaped the model, each as a link with authors and year |
| Other Public Implementations | Links to well-known implementations elsewhere |

## Elements

Links to other pages are relative paths ending in `.md`: `[Part 1](../intro/dl_intro.md)`. Link to a section with `[text](../intro/dl_intro.md#section-heading-in-lowercase-with-hyphens)`.

Math uses `$...$` inline and `$$...$$` for display. Multi-line math uses `aligned` inside the dollar signs. To number an equation, put a label in parentheses on the same line as the closing `$$`. Refer to it in the text with the `eq` role. Labels are shared across the whole site, so start each one with the page name (`eq-mlp-forward`, not `eq-forward`).

```markdown
$$
\begin{aligned}
z &= Wx + b \\
a &= \max(0, z)
\end{aligned}
$$ (eq-mlp-layer)

The second line of {eq}`eq-mlp-layer` is the ReLU.
```

Site-wide macros are `\R`, `\E`, `\L`, `\argmax`, `\argmin`. Add more under `mathjax3_config` in `docs/conf.py` when a symbol is used on more than one page.

Callout boxes are admonitions. The box title is free text. "You Should Know" is the default box, used for the point a reader must not miss.

````markdown
```{admonition} You Should Know
The body of the box. It can hold paragraphs, math, lists, and code.
```

```{admonition} Key Idea
:class: key-idea

One result worth remembering.
```
````

| Box | Syntax | Use for |
| --- | --- | --- |
| You Should Know | `{admonition} You Should Know` | Definitions and points needed later |
| Key Idea | `{admonition} Key Idea` with `:class: key-idea` | One central result |
| Note | `{note}` | Asides |
| Tip | `{tip}` | Practical advice |
| Warning | `{warning}` | Mistakes that cost time |
| Coming soon | `{admonition} Coming soon` with `:class: coming-soon` | Unwritten pages and sections |

Code blocks always name a language (`python`, `bash`, `text` for pseudocode). Code examples use PyTorch and must run as written. The site build does not execute code and the build environment has no PyTorch, so run each example in an environment that has `torch` before publishing it. If that is not possible, tell the user the example is unverified.

Figures go in `docs/_static/img/<page-name>/` and are placed with `figure` so they carry a caption. The path is relative to the page file. Prefer SVG for diagrams and PNG at twice the display size for plots.

````markdown
```{figure} ../_static/img/mlp/architecture.svg
:alt: A three-layer perceptron with two hidden layers.
:width: 80%

What the reader should notice in the figure.
```
````

A footnote backs a specific claim in the text: `[^adam]` at the claim and `[^adam]: Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2015.` as the last lines of the file. Footnotes always render at the very bottom of the page, whatever heading the definition sits under, so never put footnote definitions under a heading such as Relevant Papers: the heading would render empty. A reading list is a plain bulleted list. The same paper may appear in both.

## Voice

Write the way Spinning Up does: direct, second person, short paragraphs, one idea per paragraph. Define a term in bold at the point it is first used. State the claim, then the equation, then what each symbol means. Prefer a small concrete example over a general statement. Headings are labels, not sentences.

## Design Reference

The look comes from [aub-faf.com](https://aub-faf.com/) and is defined once, as CSS variables at the top of `docs/_static/css/faf.css`. Pages never set fonts or colors themselves. A page that needs a new visual treatment gets a class in `faf.css`, applied with `:class:` on a directive or `{.class-name}` on the line above a paragraph.

| Token | Value | Used for |
| --- | --- | --- |
| Font | `"Proxima Nova", "Cairo", Arial, Helvetica, sans-serif` | All text. No web font is loaded, matching aub-faf.com: readers with Proxima Nova installed see it, everyone else sees Arial or Helvetica |
| Code font | `ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace` | Code and inline code |
| Math | MathJax default (Computer Modern) | Equations |
| `--faf-red` | `#840132` | AUB maroon: links, section labels, callout title bars, primary button, active sidebar marker |
| `--faf-red-dark` | `#650026` | Hover and visited links, inline code text |
| `--faf-ink` | `#161616` | Headings and body text |
| `--faf-muted` | `#262626` | Secondary text |
| `--faf-soft` | `#5c5f58` | Captions, footer, breadcrumbs |
| `--faf-paper` | `#ffffff` | Page background |
| `--faf-warm` | `#f0efe9` | Sidebar, callout bodies, inline code background |
| `--faf-sage` | `#edf2ed` | Tip and "Coming soon" boxes |
| `--faf-line` | `#dedfd7` | Hairline rules and borders |

Type scale: body 17px with line height 1.7, `h1` 40px, `h2` 27px, `h3` 20.5px. Headings are weight 800 with letter spacing -0.025em. Section labels in the sidebar are 11.5px, weight 700, uppercase, letter spacing 0.15em, maroon. Corners are 3px. The content column is 840px wide and the sidebar 300px.

Diagrams and plots made for the site use the same palette: maroon `#840132` for the thing being discussed, ink `#161616` for everything else, `#f0efe9` for fills, white background, no gradients or shadows.

## Build, Preview, Publish

```bash
make setup
make serve
make check
```

`make setup` is needed once per machine. `make serve` opens a live preview at `http://localhost:8000` that reloads on save. `make check` is the gate: it builds with warnings as errors, exactly as the deploy workflow does, and writes the HTML to `docs/_build/html/`.

A page is done when `make check` passes and the rendered page has been looked at, in the preview or by opening its file under `docs/_build/html/`. The build cannot see an empty heading, a broken equation, or a figure at the wrong size; only looking can.

Publishing is a commit and a push to `main`. The GitHub Actions workflow rebuilds and deploys the site. Commit and push only when the user asks.

## Common Mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `document isn't included in any toctree` | New page not listed in `docs/index.md` | Add it to a `toctree`, or mark it `orphan: true` |
| `toctree contains reference to nonexisting document` | Path in `docs/index.md` is wrong or has an extension | Use `section/name` with no `.md` |
| "Erroneous nesting of equation structures" on the page | `\begin{align}` inside `$$` | Use `\begin{aligned}` |
| `cross-reference target not found` | Link path is not relative to the current file, or lacks `.md` | Fix the relative path |
| Page title missing from the sidebar | File has no `#` heading, or has two | Exactly one `#` heading, first in the file |
| Raw `$` signs showing in the text | Space after the opening `$` or before the closing one | Remove the spaces |
| Image not found | Path written from the repo root | Write it relative to the page file |
| A heading renders with nothing under it | Only footnote definitions were placed there | Put a bulleted list under the heading and move the footnotes to the end of the file |
| `duplicate label of equation` | Two pages use the same equation label | Prefix labels with the page name |
