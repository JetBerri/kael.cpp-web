+++
title = "Contribute"
description = "How to fix, extend and add pages to the kael.cpp documentation."
weight = 100

[extra]
kicker = "Community"
+++

This site is meant to grow. The book explains how kael is built today. The concepts section is where the community explains the ideas around it: the math, the hardware, the algorithms and the trade-offs that every deep learning framework faces. The long-term goal is a single place where an engineer can learn how modern AI systems work, from a flat buffer of floats to a model that generates text.

Every page is a Markdown file in the [site repository](https://github.com/JetBerri/kael.cpp-web). If you can write Markdown and open a pull request, you can contribute.

## Ways to contribute

| Contribution | Where | Size |
|--------------|-------|------|
| Fix a typo, a broken link or an unclear sentence | Any page, through the "Edit this page on GitHub" link | Minutes |
| Add a worked example, diagram or exercise to a chapter | `content/book/` | An evening |
| Write a concept article | `content/concepts/` | A weekend |
| Add a term to the glossary | `content/concepts/glossary.md` | Minutes |
| Request a topic you want explained | [Open a topic request](https://github.com/JetBerri/kael.cpp-web/issues/new/choose) | Minutes |
| Change the library itself | The [kael.cpp repository](https://github.com/JetBerri/kael.cpp) | Varies |

Pick an issue labelled `good first page` if you want a starting point.

## Run the site locally

The site is built with [Zola](https://www.getzola.org/), a single binary static site generator. Install it, then serve the site with live reload:

```bash
brew install zola        # macOS; see the Zola docs for Linux and Windows
git clone https://github.com/JetBerri/kael.cpp-web
cd kael.cpp-web
zola serve
```

Open `http://127.0.0.1:1111`. Run `zola check` before opening a pull request. It fails on broken internal links.

## Repository layout

```text
content/
  _index.md            landing page
  book/                one page per chapter of the book, ordered by weight
  concepts/            community articles and the glossary
  adr/                 design decision records of the library
  contributing.md      this page
templates/             Tera templates for pages, sections and shortcodes
sass/style.scss        the only stylesheet
static/                logo, icons and JavaScript
```

## Write a page

Create a Markdown file in the right section. The file starts with TOML front matter between `+++` lines.

```toml
+++
title = "Floating point for deep learning"
description = "Why float32 is the default, what bfloat16 trades away, and where rounding bites."
weight = 30
authors = ["Your Name"]

[extra]
kicker = "Concept"
math = true            # enables LaTeX with $inline$ and $$display$$
+++
```

| Field | Required | Purpose |
|-------|----------|---------|
| `title` | yes | Page title and search result title |
| `description` | yes | One sentence shown under the title, on index pages and in search |
| `weight` | yes | Order inside the section, lower first |
| `authors` | no | Names shown under the title |
| `extra.kicker` | no | Small label above the title, for example `Chapter 06` or `Concept` |
| `extra.math` | no | Set to `true` to render LaTeX with KaTeX |
| `extra.tag` | book only | The git tag in the library repository that matches the chapter |

### Link to other pages and to the code

Link to another page by its file path. Zola checks these links at build time, so a renamed file fails the build instead of producing a dead link.

```markdown
See [Views](@/book/03-views.md) and the [glossary](@/concepts/glossary.md#stride).
```

Link to a file in the library with its full GitHub URL on the `main` branch:

```markdown
[`src/autograd/engine.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/engine.cpp)
```

Pages are plain Markdown. Template syntax is disabled in content files, so C++ code such as `Tensor({{1, 2}, {3, 4}})` renders as written and needs no escaping.

### Callouts

Use GitHub alert syntax. Keep them rare: one or two per section at most.

```markdown
> [!NOTE]
> Extra context that is useful but not required to follow the text.

> [!TIP]
> A shortcut or a way to check your understanding.

> [!WARNING]
> A pitfall that produces wrong results without an error.
```

### Math

Set `extra.math = true` and write LaTeX between `$` for inline math and `$$` for display math. Put display math on its own lines.

```markdown
The gradient of the loss with respect to an input is $\partial L / \partial x_i$.

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$
```

### Diagrams

Draw diagrams as text inside a `text` code block. They render everywhere, diff cleanly and anyone can edit them.

```text
storage:  [ 0  1  2  3  4  5 ]
shape  [2, 3], strides (3, 1)
row 0 -> 0 1 2
row 1 -> 3 4 5
```

## Writing style

The audience is engineers who know how to program and want to understand how deep learning systems work inside. Write for them.

- Start from the problem. Explain why a layer or idea exists before how it works.
- Use small worked examples with real numbers. A `[2, 3]` tensor traced by hand teaches more than a paragraph of prose.
- Show real code from the library, and say which file it comes from. When an excerpt and the code disagree, the code wins.
- Use short declarative sentences and the active voice. Use the imperative mood for instructions.
- Define a term the first time you use it, or link to the glossary.
- Do not use emojis, exclamation marks, rhetorical questions or marketing words.
- Write in English. Translations may come later as a separate effort.

### Structure of a concept article

A concept article explains one idea in depth. It does not need to be tied to kael, but it should link to the kael code that implements the idea when there is some.

1. **The problem**: what goes wrong without this idea.
2. **The idea**: the mechanism, from first principles, with a worked example.
3. **In practice**: how real frameworks implement it, with code from kael where possible.
4. **Pitfalls**: what goes wrong when it is implemented badly.
5. **Further reading**: papers, books and source code, with one line on why each is worth reading.

### Structure of a book chapter

Book chapters follow the library one layer at a time. Each chapter has the same sections so readers know where to look.

1. **Why this layer exists** and the mental model.
2. **Concepts** with worked numeric examples and diagrams.
3. **Implementation**: the real types and functions, in the order you would write them.
4. **Edge cases and pitfalls**.
5. **How it is tested**.
6. **Build it yourself**: ordered steps with a checkpoint each.
7. **Check your understanding**: questions with answers hidden in a `<details>` block.

## Pull requests

- One topic per pull request. A new article and a fix to an unrelated chapter go in two pull requests.
- Use [Conventional Commits](https://www.conventionalcommits.org/) with a scope: `docs(book): add a broadcasting exercise to chapter 01`, `docs(concepts): add floating point article`.
- Run `zola check` and read your page in `zola serve` before you open the pull request.
- Technical claims must be verifiable. Link to the code, a paper or a reproducible command.

A maintainer reviews for accuracy first and style second. Expect questions; they are part of making the page correct.

## License

Contributions are released under the [Apache License 2.0](https://github.com/JetBerri/kael.cpp-web/blob/main/LICENSE), the same license as the library.
