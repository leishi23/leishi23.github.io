---
layout: post
title:  "Kramdown Math Syntax"
date:   2023-06-02
desc: "Kramdown Math Syntax"
keywords: "Linux"
categories: [Linux]
tags: [Linux]
icon: icon-html
---

# Kramdown Math Syntax
## _Linux_DebugNotes_04_

[![image](https://fossbytes.com/wp-content/uploads/2020/02/Ubuntu-18.04.4-release.jpg)](https://docs.mathjax.org/en/latest/web/start.html)

## Keypoints 
- Kramdown in Jekyll
- MathJax 
- How to write math formula in kramdown

## Kramdown in Jekyll
- Kramdown is a Markdown-superset converter
- Jekyll uses Kramdown as its default Markdown engine


## MathJax
- MathJax is a JavaScript display engine for mathematics that works in all browsers.
- Kramdown supports MathJax, but it doesn't include MathJax by default.

## How to write math formula in kramdown
- To fix Kramdown doesn't support `$$` as inline mathmatics, add the following code in `_includes/head.html`:
    ```html
    <!-- Mathjax Support -->
    <script>
    MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
        }
    };
    </script>
    <script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script>
    ```
    - `inlineMath` is used to define the inline math delimiter. It is to enable `$$` can be used in kramdown for inline mathmatical formula.

  - Math syntax in kramdown: the default math delimiters are `$$...$$` and `\[...\]` for displayed mathematics, and `\(...\)` for in-line mathematics.
- To fix matrix registers new lines issue:This is caused by rendering `\\` as `\` by default in markdown parser. And `\` is an escape character commonly used across many programming languages. You could use a `\` before every actual `\` you want to use. So if you wanted `\begin{bmatrix}` you would have to write `\\begin{bmatrix}` and if you want `\\` you need to use `\\\\`.
    - Example:
    ```markdown
    $ \\begin{bmatrix}a & b \\\\ c & d\\end{bmatrix} $
    ```
- To avoid | in $$ part. But use \vert !!!
    
- Example:
  ```markdown
    $$
    \\begin{align*}
    & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
    = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
    & (x_1, \ldots, x_n) \left( \\begin{array}{ccc}
        \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
        \vdots & \ddots & \vdots \\
        \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
        \end{array} \right)
    \left( \\begin{array}{c}
        y_1 \\
        \vdots \\
        y_n
        \end{array} \right)
    \end{align*}
    $$
    ```
    - The above code will be rendered as:
  $$
    \\begin{align*}
    & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
    = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
    & (x_1, \ldots, x_n) \left( \\begin{array}{ccc}
        \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
        \vdots & \ddots & \vdots \\
        \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
        \end{array} \right)
    \left( \\begin{array}{c}
        y_1 \\
        \vdots \\
        y_n
        \end{array} \right)
    \end{align*}
    $$
    
