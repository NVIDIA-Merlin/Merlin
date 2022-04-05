# Documentation

This folder contains the scripts necessary to build the repository
documentation. You can view the documentation at
<https://nvidia-merlin.github.io/Merlin/main/README.html>.

## Contributing to Docs

Perform the following steps to build the docs.

1. Install the documentation dependencies:

   ```shell
   python -m pip install -r docs/requirements-doc.txt
   ```

1. Run the build command:

   ```shell
   make -C docs clean html
   ```

   Remove the `-C docs` argument if you are already in the `docs/` directory.

These steps should run Sphinx in your shell and create HTML in the `build/html/`
directory.

## Preview the Changes

View the docs web page by opening the HTML in your browser. First, navigate to
the `build/html/` directory and then run the following command:

```shell
python -m http.server
```

Afterward, open a web browser and access <https://localhost:8000>.

Check that yours edits formatted correctly and read well.
