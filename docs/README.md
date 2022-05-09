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

The build for Merlin is unique because the support matrix is generated during
the build.
The build reads the `docs/data.json` file and creates several RST snippet files
in `docs/source/generated`.
The `docs/data.json` file is updated by the `docs-smx-data` GitHub workflow.

## Preview the Changes

View the docs web page by opening the HTML in your browser. First, navigate to
the `build/html/` directory and then run the following command:

```shell
python -m http.server
```

Afterward, open a web browser and access <https://localhost:8000>.

Check that yours edits formatted correctly and read well.

## Tests

```shell
python -m pytest docs
```
...or...

```shell
coverage run -m pytest -v docs && coverage report -m
```

## Handy notes

### Remove a field from the JSON

In the following case, the `cuparse` key is a mistake and should have been
`cusparse`.  The following command removes the mistake from the JSON:

```shell
jq 'walk(if type == "object" then del(.cuparse) else . end)' < data.json > x
```

### View a container for a release

```shell
jq '.["nvcr.io/nvidia/merlin/merlin-inference"]["22.03"]' < ../docs/source/data.json
```

### List the containers and releases

```shell
jq '. | map_values(keys) ' < docs/source/data.json
```
