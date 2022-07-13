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

1. Start Python in a container:

  ```shell
  docker run --rm -it -v $(pwd):/workspace -w /workspace --network=host \
    python:3.8-buster@sha256:ccc66c06817c2e5b7ecd40db1c4305dea3cd9e48ec29151a593e0dbd76af365e bash
  ```

1. Install dependencies in the container:

  ```shell
  python -m pip install pip==22.0.4 setuptools==59.4.0 wheel
  python -m pip install -r docs/requirements-doc.txt
  ```

  > Pip is frozen at 22.0.4 because that version is specified
  > in `test_extractor.py`.  Update the test if you update
  > the version here and in `requirements-doc.txt`.

1. Run the tests:

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
jq '.["nvcr.io/nvidia/merlin/merlin-hugectr"]["22.03"]' < ../docs/source/data.json
```

### List the containers and releases

```shell
jq '. | map_values(keys) ' < docs/source/data.json
```

## About the support matrix

Documenting the support matrix is a three part process:

* Extract the data from the containers.
* Convert the data into RST.
* Build the docs, as normal.

The first part is handled by the `docs/extractor.py` script.
Unless you have all the containers downloaded, it's best to run
this script on a machine with high bandwidth and with automation, such as Blossom.
Check in the team's Blossom instance for a `docs-smx-data` job.

By default, the script pulls each container (the containers are listed in the
script itself) and uses the YY.MM label based on the current data.
The script attempts to extract information from `Pip`, the environment for
the container, environment variables in the container, and so on.
After the data is extracted, the script updates the `data.json` file.

At the start of the documentation build, Sphinx runs the `docs/smx2rst.py` script.
This script reads the `data.json` file and creates an RST file in
`docs/source/generated` for each container.
Each file includes tables, by year, that are based on the data in the JSON file.

The `docs/source/support_matrix` directory has an RST that corresponds to one of
the generated files.
You can add text into those files to indicate information like TensorFlow is
not installed in the inference container for TensorFlow models.
