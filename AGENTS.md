# Spinning Up in Deep Learning

This repository is the website for Spinning Up in Deep Learning, a deep learning course from AUB Frontier AI Fellows (AUB–FAF). It is modelled on OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com/).

**The standard for every change: a page here must look and read as if it belonged to Spinning Up.** Only the subject (deep learning), the name, and the logo are different.

The rules below, the files in `templates/`, and `docs/etc/style-guide.md` are the reference, in that order. When they do not cover a choice and you have network access, open the matching Spinning Up page, or its source under [openai/spinningup/docs](https://github.com/openai/spinningup/tree/master/docs), and do what it does.

## Commands

```bash
make setup
make serve
make check
make shot PAGE=intro/neural_networks
```

- `make setup` creates `.venv` and installs the build tools. Run it once per machine.
- `make serve` opens a live preview at `http://localhost:8000`.
- `make check` is the gate. It runs the template checker (`tools/lint_content.py`) and then builds the site with warnings as errors. The deploy runs the same two steps, so a change that fails `make check` will not publish.
- `make shot PAGE=<section>/<file name>` saves a screenshot of a built page to `docs/_build/shots/` and prints its path. Add `HEIGHT=6000` for a long page.

## Where Things Are

| What | Where |
| --- | --- |
| Pages | `docs/`, one folder per sidebar section |
| Landing page and sidebar order | `docs/index.md` |
| Page and section templates | `templates/` |
| Every element, rendered, with its source | `docs/etc/style-guide.md` |
| Step-by-step mechanics (new pages, renames, figures, element syntax, fonts and colors) | `.claude/skills/add-content/SKILL.md` |
| Template checker | `tools/lint_content.py` |
| Styling, ported from Spinning Up | `docs/_static/css/modify.css` |

## The Usual Task: Typesetting Notes

The maintainer adds the course one piece at a time by handing over notes for a topic: pasted text, a PDF, photos, or LaTeX. Turn those notes into a section of the site that follows the template.

1. **Find where it goes.** The five parts of the course are the files in `docs/intro/`. Each topic is a `##` heading with `*Coming soon.*` under it. Match the notes to a heading. If no heading matches, ask the maintainer where the topic belongs before adding one.
2. **Read the template.** `templates/section.md` is a complete example of one topic. `docs/etc/style-guide.md` shows the syntax of every element.
3. **Write the section.** Replace the `*Coming soon.*` line under the heading with the content. Leave the heading text and the other sections as they are.
4. **Stay faithful to the notes.** Every definition, equation, example, and claim in the notes appears in the section, in the order the notes give them. A point the notes make twice appears once, where it fits best. Rewrite fragments into full sentences, fix typos, and group related points under `###` subsections.
5. **Add only what the template requires.** That means the section's opening sentences, the sentences that connect one point to the next, and a short definition of any symbol the reader has not met. Do not add examples, numbers, code, or figures that the notes lack unless the maintainer asks for them. List every sentence of substance that did not come from the notes in your reply, so the maintainer can accept or remove it.
6. **Flag what looks wrong.** If the notes contain an error, such as a wrong sign or a mismatched dimension, or two statements that disagree, do not silently change it and do not silently copy it. Typeset what you believe is correct and say in your reply what the notes had and what you wrote.
7. **Keep the page consistent.** See the next section.
8. **Check.** Run `make check`. Then run `make shot` on the page and look at the image: equations render, figures are the right size, no heading is empty. Math is drawn in the browser, so reading the HTML file does not tell you whether an equation renders.
9. **Report.** Tell the maintainer which section was written, what you added beyond the notes, and anything you flagged.

Figures in the notes are redrawn as SVG in the site palette, or used as supplied when the maintainer provides the image file. They go in `docs/_static/img/<page-file-name>/`. Do not copy figures, text, or code from papers, books, or other courses; describe the idea in the site's own words and cite the source with a footnote.

### While a Part Is Half Written

Sections arrive in any order, so a part page spends most of its life half written. Its layout, top to bottom, is: title, table of contents, `Coming soon` box, opening paragraph, sections. `templates/part.md` shows it.

- **Opening paragraph.** The first time any section of a part is written, whichever heading it is, add the part's opening paragraph. Write the list of what the part covers from its section headings. Do not invent prerequisites: say what the reader should already know only if the notes or the maintainer say so. Report the opening paragraph as something you added.
- **Symbols from unwritten sections.** If your section needs a symbol or term that an earlier, unwritten section would normally introduce, define it in one short clause and move on. Bold a term where the site first defines it as the site stands today.
- **Links.** Link back to an earlier section only once it is written. Never link to a section that still says `*Coming soon.*`.
- **Finishing.** When the last section is written, delete the `Coming soon` box and add a `## Recap` section at the end. The checker requires both.

## House Style

This is what "like Spinning Up" means in practice. The checker enforces the mechanical parts; the rest is judgment.

### Page Shape

- One `#` title, then the table of contents, then the page. The table of contents lists the page title and its `##` topics, as Spinning Up's does.
- A part page opens with a short welcome paragraph and a bulleted list of what it covers, and ends with a `## Recap`.
- `##` for topics, `###` for the parts of a topic, `####` at most. Levels are never skipped.
- Headings are short labels in Title Case with no numbers and no closing punctuation: "Vanishing and Exploding Gradients", not "1.7 Vanishing / exploding gradients:". An optional section says so in its heading: "(Optional) Formalism".

### Section Shape

- Open with what the topic is and why the reader needs it now. One or two sentences.
- Build one step at a time. Each paragraph carries one idea.
- A new term is **bold** at the point where it is defined, and only there.
- A sentence leads into each equation, and the sentence after it explains any symbol the reader has not met yet: "we replace it by", then the equation, then "where $c$ is the threshold". Equations are part of the sentence and take its punctuation.
- When the notes give a concrete case and a general rule, the concrete case comes first.
- A code example is introduced by a bold label: "**Example: Clipping in a Training Step.**"
- End a topic when it is finished. No summary paragraph per section; the part's Recap does that job.

### Voice

- Write as "we" speaking to "you", the way Spinning Up does: "we replace it by", "you can skip this example and come back to it later".
- Plain and direct. Prefer the short word. Say the thing instead of announcing that you will say it.
- Friendly but not chatty. No jokes that need explaining, no exclamation marks outside the welcome line.

### Math

- Inline math in `$...$`, display math in `$$...$$`, multi-line math in `aligned` inside the dollar signs, piecewise functions in `cases`.
- Number an equation only when the text refers back to it. A label is `eq-`, then the page's file name with underscores written as hyphens, then a short name: `$$ (eq-improving-networks-adam)` in `improving_networks.md`.
- Use the same symbol for the same quantity on every page. Before introducing a symbol, search `docs/` for how written sections typeset it.

### Notation

The notes use plain-text shorthand. Typeset it as follows, on every page.

| In the notes | Typeset as | Meaning |
| --- | --- | --- |
| `W[l]`, `b[l]`, `z[l]`, `a[l]` | $W^{[l]}$, $b^{[l]}$, $z^{[l]}$, $a^{[l]}$ | A quantity of layer $l$ |
| `x(i)`, `y(i)` | $x^{(i)}$, $y^{(i)}$ | Training example $i$ |
| `x<t>`, `a<t>` | $x^{\langle t \rangle}$, $a^{\langle t \rangle}$ | Time step $t$ of a sequence |
| `X{t}`, `Y{t}` | $X^{\{t\}}$, $Y^{\{t\}}$ | Mini-batch $t$ |
| `m`, `n_x`, `L` | $m$, $n_x$, $L$ | Number of examples, of input features, of layers |
| `y^`, `yhat` | $\hat{y}$ | The model's prediction |
| `g(z)` | $g(z)$ | An activation function applied to $z$ |
| `dW`, `db` | $dW$, $db$ | The gradient of the cost with respect to $W$, $b$ |
| `J`, `L(y^, y)` | $J$, $\mathcal{L}(\hat{y}, y)$ (written `\L` on the site)  | The cost over the training set, the loss on one example |

If the notes use notation that is not in this table, typeset it as the notes write it and add a row here, so that later sections match.

### Boxes

- "You Should Know" is the default box. Use it for a convention the reader will meet elsewhere, a caveat, a common mistake, or a pointer to further reading. Write it as ` ```{admonition} You Should Know `.
- A box may carry its own title when it defines a named thing, as Spinning Up does with "Categorical Policies".
- `{note}`, `{tip}`, and `{warning}` exist but are rare. A page with a box in every section has too many.

### Code

- PyTorch. Short enough to read without scrolling. Complete enough to run as written.
- Every code block names its language: `python`, `bash`, or `text` for pseudocode.
- The build has no PyTorch and does not run the examples. Run each one where `torch` is installed, or tell the maintainer it is unverified.

### Links and Citations

- Links to other pages are relative paths ending in `.md`. Link to a section with `#the-heading-in-lowercase-with-hyphens`.
- Link the phrase that names the thing, never "here" or "this link".
- A paper is cited with a footnote whose definition sits at the very end of the file.

## What Not to Do

- Do not change the look. Fonts, colors, and layout live in `docs/_static/css/modify.css`, `docs/conf.py`, and `docs/_templates/`. Leave them alone unless the maintainer asks for a design change.
- Do not put raw HTML, inline styles, CSS classes, or Markdown image syntax in a page. The checker rejects them.
- Do not add a new directive, Sphinx extension, or Python dependency to make one page work. Use the elements in the style guide.
- Do not edit `tools/lint_content.py` to make a page pass. Fix the page. Change the checker only when the maintainer changes the template.
- Do not leave `TODO` markers, empty headings, or half-written sections in a page. A section is either written or still says `*Coming soon.*`.

## Publishing

The site deploys from `main` through `.github/workflows/deploy.yml`. Publishing is a commit and a push to `main`. Do it when the maintainer asks you to publish, or has told you to publish each section as it is finished. Write the commit message as one line that says what was added, such as "Write the Dropout section of Part 2".
