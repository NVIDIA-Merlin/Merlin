# Merlin Changelog

## [23.05]

### NVTabular

This is the last NVTabular release that will contain the `nvtabular.inference` package, which has been deprecated for quite some time and replaced by the Merlin Systems library. Starting in the 23.06 release, we recommend using Merlin Systems to serve models and NVT workflows.

## [23.04]

Changes since the 23.02 release.

The major focus of this release was broader support for training and serving session-based models across all Merlin libraries.

### NVTabular

#### Added

* Ability to use Workflows to locally transform single Pandas/cuDF dataframes without using Dask ([#1777](https://github.com/NVIDIA-Merlin/NVTabular/pull/1777))
* The `Categorify` operator now supports int16 dtypes when serving Workflows ([#1798](https://github.com/NVIDIA-Merlin/NVTabular/pull/1798))
* NVTabular can now be used in our containers without a GPU present.

#### Major Changes

* N/A

#### Deprecated / Removed

* The code for exporting Workflows and models to be served with Triton that’s currently in `nvtabular.inference` has reached end-of-life and will be removed in the next release. We recommend migrating to use Merlin Systems, which provides similar functionality.

### Dataloader

#### Added

* An operator for padding ragged list features after they’re loaded from disk and before they’re provided to a model ([#125](https://github.com/NVIDIA-Merlin/dataloader/pull/125))
* An operator for loading pre-trained embeddings from disk and merging them into training batches on the fly ([#138](https://github.com/NVIDIA-Merlin/dataloader/pull/138))

#### Major Changes

* We’ve changed the way the dataloaders use DLpack for moving data from dataframes to deep learning frameworks, and now transfer each batch individually instead of transferring large chunks of data. This allows us to unify the dataloader implementations, providing consistent performance and common operators across all frameworks. ([#111](https://github.com/NVIDIA-Merlin/dataloader/pull/111), [#121](https://github.com/NVIDIA-Merlin/dataloader/pull/121))
* The output format for ragged list features has been changed from a (values, offsets)  tuple to two separate keys `col_name__values` and `col_name__offsets` ([#101](https://github.com/NVIDIA-Merlin/dataloader/pull/101))

#### Deprecated / Removed

* Sparse tensors are no longer supported as an output format for ragged list features ([#103](https://github.com/NVIDIA-Merlin/dataloader/pull/103))

### Models

#### Added

* [Quick-start for Ranking](https://github.com/NVIDIA-Merlin/Merlin/blob/main/examples/quick_start/ranking.md) - A new guide on how to preprocess data and, build, train, evaluate and hypertune Ranking models with Merlin, including best practices to improve models accuracy ([#915](https://github.com/NVIDIA-Merlin/Merlin/pull/915))
* We have added support for Tensorflow 2.11 and 2.12 ([#1016](https://github.com/NVIDIA-Merlin/models/pull/1016), [#1040](https://github.com/NVIDIA-Merlin/models/pull/1040))
* We have introduced `SOKEmbedding` class that leverages HugeCTR Sparse Operations Kit for model parallelism ([#863](https://github.com/NVIDIA-Merlin/models/pull/863)).

#### Major Changes

* Publishing to Anaconda beginning with 23.04 https://anaconda.org/nvidia/merlin-models.
* We have introduced a new design of the transformer API that simplifies the high-level transformer API and also fixes issues related to Causal LM (#1022).

#### Bug Fixes

* Fixes `model.batch_predict()` which was not working for ranking models with the latest output layer API (ModelOuput) ([#1052](https://github.com/NVIDIA-Merlin/models/pull/1052))
* Refactor/fix of sampled softmax to add logQ correction, which is important for better accuracy, as logQ correction avoids over penalizing popular items for being sampled as negatives more often ([#1051](https://github.com/NVIDIA-Merlin/models/pull/1051)).

#### Deprecated / Removed

* We have removed the PyTorch backend in preparation for new and improved PyTorch support  (#1020).

### Transformers4Rec

#### Added

* Added sampled softmax for faster training, as it computes loss on sampled items instead of all items. It is also possible to get a better accuracy using sampled softmax, check the PR to see some benchmark results ([#671](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/671)).

#### Major Changes

* Now has a hard dependency on Merlin Models verison 23.04 or greater.

#### Deprecated / Removed

* Began migration of Tags from `merlin_std_lib` module to the `merlin.schema` module provided by `merlin-core`. This is the first step in an effort to completely remove the `merlin_std_lib` module, which will happen over several releases.

### Systems

#### Added
* Serving session-based models from Merlin Models and Transformers4Rec as part of Ensembles ([#299](https://github.com/NVIDIA-Merlin/systems/pull/299) and many others)
* Using the dataloader padding and pre-trained embedding ops in Ensembles ([#329](https://github.com/NVIDIA-Merlin/systems/pull/329), [#330](https://github.com/NVIDIA-Merlin/systems/pull/330))
* Exporting Ensembles to Triton with kind AUTO, so that they work on both CPU and GPU environments without needing to re-export ([#321](https://github.com/NVIDIA-Merlin/systems/pull/321))

#### Major Changes

* Systems now makes use of shape information captured in schemas by other Merlin libraries in order to create and validate Triton configuration files. ([#293](https://github.com/NVIDIA-Merlin/systems/pull/293))
* Ragged list features are now represented in requests/responses with two separate values and offsets arrays ([#296](https://github.com/NVIDIA-Merlin/systems/pull/296))

#### Deprecated / Removed

* A validation and error concerning extra columns produced by Ensemble operators that were never used downstream. We’ve made internal changes in Systems to allow this, but it’s still not recommended. ([#306](https://github.com/NVIDIA-Merlin/systems/pull/306))

### Core
#### Added
* A tag for labeling pre-trained embedding features in order to distinguish them from continuous and categorical features ([#239](https://github.com/NVIDIA-Merlin/core/pull/239))
* An option to select by any or all tags when using “Schema.select_by_tag()” ([#94](https://github.com/NVIDIA-Merlin/core/pull/94))
* A `row_group_size` argument on `Dataset.to_parquet()` that allows controlling the maximum number of rows in a single Parquet row-group ([#218](https://github.com/NVIDIA-Merlin/core/pull/218))
#### Major Changes

* Merlin Core and many other Merlin libraries now use a `TensorTable` abstraction to package together sets of NumPy/CuPy/Tensorflow/Torch arrays or tensors into very minimal but dataframe-like objects that can be used with many Merlin operators, like those from the dataloaders and systems libraries. ([#230](https://github.com/NVIDIA-Merlin/core/pull/230) and many subsequent PRs throughout Merlin)
* Strict dtype checking when running Merlin DAGs (like NVT Workflows) is no longer the default. This should reduce the number of dtype discrepancy errors encountered by users in cases where they don’t ultimately matter, but may allow some cases that don’t work with downstream operators or libraries. ([#268](https://github.com/NVIDIA-Merlin/core/pull/268)) 

#### Deprecated / Removed

* N/A

### Other / Misc

#### Deprecated / Removed
* Removed `distributed-embeddings` from the 23.04 release. This will be added back in future releases. It is still possible to build the container by setting the `INSTALL_DISTRIBUTED_EMBEDDINGS` argument to `true`. ([Merlin/#908](https://github.com/NVIDIA-Merlin/Merlin/pull/908))
