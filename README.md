# Course website template

**Find instructions at [sib-swiss.github.io/gh-pages-training/](https://sib-swiss.github.io/gh-pages-training/).**

This website is generated with [MkDocs](https://www.mkdocs.org/), with the theme [Material](https://squidfunk.github.io/mkdocs-material/).

To host it locally, install mkdocs-material: 

```bash
pip install mkdocs-material
```

Fork this repository and clone it to your local computer. Then, make the repository your current directory and type:

```bash
mkdocs serve
```

To host it locally.

Check it out with your browser at [http://localhost:8000/](http://localhost:8000/).

If you are ready to host it on GitHub, you can run: 

```sh
mkdocs gh-deploy
```

This will generate a webpage at:

https://yourname.github.io/reponame

After that, the workflow specified at `.github/workflows/render_page.yml` will rebuild the website after you push to the main branch. 

More documentation can be found on the [MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/).
