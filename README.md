# kael.cpp-web

Source of the [kael.cpp website](https://kaelcpp.dev/): the book that rebuilds the [kael](https://github.com/JetBerri/kael.cpp) deep learning framework one layer at a time, design decision records, and a knowledge base, open to contributions, on how deep learning systems work.

## Build

The site uses [Zola](https://www.getzola.org/) 0.23.

```bash
zola serve    # live preview at http://127.0.0.1:1111
zola check    # fails on broken internal links
zola build    # writes the site to public/
```

Pushes to `main` build and deploy to GitHub Pages through `.github/workflows/deploy.yml`. Pull requests run the same build without deploying.

## Layout

```text
content/book/        one page per chapter of the book
content/concepts/    community articles and the glossary
content/adr/         design decision records
content/contributing.md
templates/           Tera 2 templates and components
sass/style.scss      stylesheet
static/              logo, icons and JavaScript
```

## Contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md). The full guide, with the page format and writing style, is on the site at [/contributing](https://kaelcpp.dev/contributing/).

## License

[Apache License 2.0](LICENSE), the same license as the library.
