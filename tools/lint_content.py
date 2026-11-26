import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = [ROOT / "docs", ROOT / "templates"]
SKIP_DIRS = {"_build", "_static", "_templates"}

MARKDOWN_DIRECTIVES = {"admonition", "note", "tip", "warning"}
OPAQUE_DIRECTIVES = {"contents", "toctree", "figure", "image", "math", "code-block", "list-table", "table"}
IMAGE_DIRECTIVES = {"figure", "image"}
PLACEHOLDER = "*Coming soon.*"
TOC_OPEN = "```{contents} Table of Contents"

FENCE = re.compile(r"^(\s*)(`{3,}|:{3,})\s*(.*?)\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
DIRECTIVE = re.compile(r"^\{([a-zA-Z0-9_:-]+)\}\s*(.*)$")
OPTION = re.compile(r"^\s*:([a-zA-Z_-]+):")
INLINE_CODE = re.compile(r"`[^`]*`")
RAW_HTML = re.compile(r"</?(div|span|font|img|br|center|table|tr|td|iframe|video|style|script|h[1-6])(?=[\s>/])", re.I)
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
BAD_ENV = re.compile(r"\\begin\{(align\*?|equation\*?|eqnarray\*?|gather\*?|multline\*?)\}")
EQ_LABEL = re.compile(r"^\$\$\s*\(([^)]+)\)\s*$")
LEFTOVER = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b|lorem ipsum", re.I)
NUMBERED = re.compile(r"^\d+(\.\d+)*[.)]?\s")
FILE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*\.md$")


def collect(targets):
    files = []
    for target in targets:
        if target.is_file():
            files.append(target)
            continue
        for path in sorted(target.rglob("*.md")):
            if not SKIP_DIRS.intersection(path.relative_to(target).parts):
                files.append(path)
    return files


def lint(path):
    problems = []

    def report(line, message):
        problems.append((line, message))

    lines = path.read_text(encoding="utf-8").split("\n")
    in_intro = path.parent.name == "intro" and path.parent.parent.name == "docs"
    is_landing = path.name == "index.md" and path.parent.name == "docs"

    label_prefix = "eq-" + path.stem.replace("_", "-") + "-"

    if not FILE_NAME.match(path.name):
        report(1, f"file name '{path.name}' must be lowercase letters, digits, underscores or hyphens")

    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                start = i + 1
                break

    stack = []
    headings = []
    pending_image = None
    placeholders = 0
    top_box = False
    first_content = None

    for index in range(start, len(lines)):
        number = index + 1
        raw = lines[index]
        stripped = raw.strip()

        if "\t" in raw:
            report(number, "tab character; indent with spaces")

        fence = FENCE.match(raw)
        if fence:
            marker, info = fence.group(2), fence.group(3)
            if stack and stack[-1]["char"] == marker[0] and len(marker) >= stack[-1]["length"] and not info:
                closed = stack.pop()
                if closed["name"] in IMAGE_DIRECTIVES and not closed["alt"]:
                    report(closed["line"], f"{{{closed['name']}}} needs an :alt: option describing the image")
                continue
            if stack and stack[-1]["kind"] == "code":
                continue
            if first_content is None:
                first_content = (number, stripped)
            if not info:
                report(number, "code block has no language; write ```python, ```bash, or ```text")
                stack.append({"char": marker[0], "length": len(marker), "kind": "code", "name": "", "line": number, "alt": False})
                continue
            directive = DIRECTIVE.match(info)
            if directive:
                name, argument = directive.group(1), directive.group(2)
                if name in MARKDOWN_DIRECTIVES:
                    kind = "markdown"
                elif name in OPAQUE_DIRECTIVES:
                    kind = "code"
                else:
                    kind = "code"
                    allowed = ", ".join(sorted(MARKDOWN_DIRECTIVES | OPAQUE_DIRECTIVES))
                    report(number, f"directive {{{name}}} is not part of the template; allowed: {allowed}")
                if name == "admonition" and not argument:
                    report(number, "{admonition} needs a title, for example: ```{admonition} You Should Know")
                if name == "admonition" and argument == "Coming soon" and len(headings) == 1 and not stack:
                    top_box = True
                stack.append({"char": marker[0], "length": len(marker), "kind": kind, "name": name, "line": number, "alt": False})
            else:
                stack.append({"char": marker[0], "length": len(marker), "kind": "code", "name": "", "line": number, "alt": False})
            continue

        if stack:
            option = OPTION.match(raw)
            if option:
                if option.group(1) == "alt":
                    stack[-1]["alt"] = True
                if option.group(1) == "class":
                    report(number, ":class: is not allowed; pages do not set their own styling")
            if stack[-1]["kind"] == "code":
                continue

        if not stripped:
            continue

        if first_content is None:
            first_content = (number, stripped)

        text = INLINE_CODE.sub("", raw)

        heading = HEADING.match(raw)
        if heading and not stack:
            level, title = len(heading.group(1)), heading.group(2)
            if headings and level > headings[-1][0] + 1:
                report(number, f"heading jumps from level {headings[-1][0]} to level {level}")
            if level > 4:
                report(number, "headings deeper than #### are not used")
            if title and title[-1] in ".:;,":
                report(number, "heading ends with punctuation")
            plain = INLINE_CODE.sub("x", title)
            if plain and plain[0].isalpha() and plain[0].islower():
                report(number, "heading must start with a capital letter (Title Case)")
            if in_intro and NUMBERED.match(title):
                report(number, "section headings are not numbered")
            if level == 2 and any(l == 2 and t == title for l, t, _ in headings):
                report(number, f"duplicate section heading '{title}'")
            headings.append((level, title, number))
            continue
        if heading and stack:
            report(number, "heading inside a callout box")
            continue

        if stripped == PLACEHOLDER:
            placeholders += 1

        if RAW_HTML.search(text) or "style=" in text:
            report(number, "raw HTML or inline styling; use Markdown and the documented directives")
        if "![" in text:
            report(number, "Markdown image syntax; use the {figure} directive so the image has a caption and alt text")
        if stripped.startswith("{.") or re.search(r"\]\([^)]*\)\{\.", text):
            report(number, "custom CSS classes are not allowed on pages")
        if BAD_ENV.search(raw):
            report(number, "use \\begin{aligned} inside $$ ... $$ instead of align or equation environments")
        label = EQ_LABEL.match(stripped)
        if label and not label.group(1).startswith(label_prefix):
            report(number, f"equation label '{label.group(1)}' must start with '{label_prefix}' (eq- and this page's file name)")
        if LEFTOVER.search(text):
            report(number, "leftover marker (TODO, TBD, FIXME, lorem ipsum)")
        for target in LINK.findall(text):
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            file_part = target.split("#", 1)[0]
            if not file_part.endswith(".md"):
                report(number, f"link '{target}' must be a relative path ending in .md")
            elif not (path.parent / file_part).resolve().is_file() and "templates" not in path.parts:
                report(number, f"link '{target}' points to a file that does not exist")

    for block in stack:
        report(block["line"], "block is never closed")

    h1 = [h for h in headings if h[0] == 1]
    h2 = [h for h in headings if h[0] == 2]
    if len(h1) != 1:
        report(1, f"page must have exactly one # title, found {len(h1)}")
    elif first_content and first_content[0] != h1[0][2]:
        report(first_content[0], "the # title must be the first thing on the page")

    has_toc = any(line.strip() == TOC_OPEN for line in lines)
    if has_toc:
        toc_line = next(i for i, line in enumerate(lines) if line.strip() == TOC_OPEN)
        if toc_line + 1 >= len(lines) or lines[toc_line + 1].strip() != ":depth: 2":
            report(toc_line + 1, "table of contents must be followed by ':depth: 2'")
        between = [l for l in lines[(h1[0][2] if h1 else 0):toc_line] if l.strip()]
        if between:
            report(toc_line + 1, "table of contents must come directly under the # title")
    needs_toc = (in_intro or len(h2) >= 2) and not is_landing
    if needs_toc and not has_toc:
        report(1, "page with sections must open with ```{contents} Table of Contents and :depth: 2 under the title")

    if in_intro:
        if top_box and placeholders == 0:
            report(1, "every section is written; delete the 'Coming soon' box at the top of the page")
        if placeholders and not top_box:
            report(1, "page still has '*Coming soon.*' sections; keep the 'Coming soon' box under the table of contents")
        if not top_box and h2 and h2[-1][1] != "Recap":
            report(h2[-1][2], "a finished part ends with a '## Recap' section")

    return problems


def main(arguments):
    targets = [pathlib.Path(a).resolve() for a in arguments] or [t for t in DEFAULT_TARGETS if t.exists()]
    files = collect(targets)
    total = 0
    for path in files:
        for line, message in sorted(lint(path)):
            try:
                shown = path.relative_to(ROOT)
            except ValueError:
                shown = path
            print(f"{shown}:{line}: {message}")
            total += 1
    if total:
        print(f"\n{total} problem(s) in {len(files)} file(s). Pages must follow templates/ and AGENTS.md.")
        return 1
    print(f"{len(files)} pages follow the template.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
