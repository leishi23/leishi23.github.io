# leishi23.github.io

Personal website of **Lei Shi** — PhD researcher in robotics at UW–Madison
(motion planning, control, reinforcement learning). Built with
[Jekyll](https://jekyllrb.com/) and hosted on GitHub Pages.

The landing page is a custom editorial design; the blog keeps the original
[Jalpc](https://github.com/jarrekk/Jalpc) structure, restyled to match.

## Structure

- `index.html` + `_includes/landing/*` — landing page, data-driven from `_data/index/*`.
- `static/css/landing.scss` — landing design system.
- `static/css/blog.scss` — blog cohesion layer (repaints the Bootstrap/Jalpc blog).
- `static/js/landing.js` — landing interactions (scroll reveal, nav, mobile menu).
- `_posts/` — blog posts (Markdown, kramdown + Rouge highlighting, MathJax for math).
- `_layouts/`, `_includes/`, `_sass/` — templates and legacy theme styles.
- `_config.yml` — site + author config. `author.cv` is the single source for the CV link.

## Local development

```shell
bundle install
bundle exec jekyll serve
# then open http://localhost:4000/
```

## Credits

Landing page: custom. Blog scaffold: [Jalpc](https://github.com/jarrekk/Jalpc) by jarrekk (MIT License).
