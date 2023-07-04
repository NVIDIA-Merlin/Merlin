# Merlin Changelog

<!--  This is a template to copy/paste when starting the changelog for a new release:

## [release]

### NVTabular

#### Major Changes

#### Added

#### Deprecated/Removed

#### Fixed Bugs

### Models

#### Major Changes

#### Added

#### Deprecated/Removed

#### Fixed Bugs

### Transformers4Rec

#### Major Changes

#### Added

#### Deprecated/Removed

#### Fixed Bugs

### Core

#### Major Changes

#### Added

#### Deprecated/Removed

#### Fixed Bugs

### Systems

#### Major Changes

#### Added

#### Deprecated/Removed

#### Fixed Bugs

### Dataloader

#### Major Changes

#### Added

#### Deprecated/Removed

#### Fixed Bugs
-->

## [23.06]

### NVTabular

#### Major Changes

* Moved some functionality from NVTabular to `merlin-core`, but left alias implace for import backwards compatibility. Some examples are `LambdaOp`, `AddMetadataOp`, `StatOperator`, `WorkflowNode`, and others. [#1823](https://github.com/NVIDIA-Merlin/NVTabular/pull/1823), [#1825](https://github.com/NVIDIA-Merlin/NVTabular/pull/1825)
* Updated `Categorify` to correctly handle nulls [#1836](https://github.com/NVIDIA-Merlin/NVTabular/pull/1836).

#### Added

* Added support for retrieving subworkflows using get_subworkflow API. Returns a subgraph wrapped in a new workflow object. [#1842](https://github.com/NVIDIA-Merlin/NVTabular/pull/1842)

#### Deprecated/Removed

* Removed the `nvtabular.inference` module. This functionality now exists in `merlin-systems` [#1822](https://github.com/NVIDIA-Merlin/NVTabular/pull/1822)
#### Fixed Bugs

### Models

#### Added

* Add support of transformer-based retrieval models [#1128](https://github.com/NVIDIA-Merlin/models/pull/1128)

### Merlin

#### Added

* Improvements in Quick-start for ranking example [#1014](https://github.com/NVIDIA-Merlin/Merlin/pull/1014)
  * In `preprocessing.py`, added support to target encoding features, configurable through these new CLI arguments: `--target_encoding_features`, `--target_encoding_targets`, `--target_encoding_kfold`, `--target_encoding_smoothing`.
  * In `ranking.py`: added support to select some columns to keep (`--keep_columns`) or remove (`--ignore_columns`) from at dataloading / training / evaluation.

#### Fixed Bugs

* Fixed in Quick-start for ranking example [#1017](https://github.com/NVIDIA-Merlin/Merlin/pull/1017):
  * Fixed `preprocessing.py`, which was not standardizing and tagging continuous columns properly
  * Fixed Wide&Deep and DeepFM models to use the updated API

### Transformers4Rec

#### Added

* Improved docstring coverage [#706](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/706)

#### Fixed Bugs

* Add support for providing a scalar cut-off in metrics, and Fix recall@1 that results higher than the upper cut-offs sometimes. [#720](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/720)
* Fix the CLM performance mismatch between model evaluation and manual inference [#723](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/723)
* Fixed OOM issues when evaluating/predicting [#721](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/721)
  * API breaking notice: This fix changes the default output of `trainer.predict()` API, that returns a `PredictionOutput` object with a predictions property. Before this change, when the `predict_top_k` option was not set (default) the predictions property was as 2D tensor (batch size, item cardinality) with the scores for all the items. As now we set `T4RecTrainingArguments.predict_top_k` by default, the predictions property returns a tuple with `(top-100 predicted item ids, top-100 prediction scores)`.

### Core

#### Major Changes

* Merged NVTabular Operator base class with Base Operator in core. 

#### Added

* Migrated some operators from NVTabular to core, allowing use in `merlin-systems`. (i.e. `LambdaOp` - changed to user defined function (UDF) and add metadata operator).
* Created subgraph operator to allow for recall and use of parts of a graph

### Systems

#### Added

* Added Test cases to debut functional support for core operators in systems ensembles
* Added API to retrieve sub ensembles. 

## [23.05]

### NVTabular

#### Added

* Support for using the `int8` dtype with NVT's `Categorify` operator at inference time ([#1818](https://github.com/NVIDIA-Merlin/NVTabular/pull/1818))

#### Deprecated/Removed

* This is the last NVTabular release that will contain the `nvtabular.inference` package, which has been deprecated and slated for removal for quite some time. Starting in the 23.06 release, we recommend using Merlin Systems to serve models and NVT workflows.

### Models

#### Added

* Added support in the Models library to use pre-trained embeddings provided by the `EmbeddingOperator` transform from Merlin dataloader. Those embeddings are non-trainable, and can be easily normalized, projected to another dim and combined with the other categorical and continuous features. ([#1083](https://github.com/NVIDIA-Merlin/models/pull/1083))
* Added a `LogLossMetric`, a Keras callback to track throughput (`ExamplesPerSecondCallback`) and a class to manage metrics logging to Weights&Biases (`WandbLogger`) ([#1085](https://github.com/NVIDIA-Merlin/models/pull/1085))
* Extended `ContrastiveOutput` to support sequential encoders ([#1086](https://github.com/NVIDIA-Merlin/models/pull/1086)) - adds support for negative sampling to the `ContrastiveOutput` class for session-based models where the query encoder returns a 3-D ragged tensor.

#### Bugs

* Change tf.keras.optimizers.Adagrad() to tf.keras.optimizers.legacy.Adagrad() ([#1098](https://github.com/NVIDIA-Merlin/models/pull/1098))

### Transformers4Rec

#### Added

* Added topk arg to return topk items and scores at inference step - added functionality for returning topk most relevant (with the highest scores) item ids for NextItemPrediction task ([#678](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/678))
* Added Transformers Torch Extras to install requirements ([#699](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/678))

#### Bugs

* Fixed the projection layer when using weight tying and dim from Transformer output and item embedding differs ([#689](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/689))

#### Deprecated / Removed

* The legacy inference api was removed from the example and was replaced by Merlin Systems api ([#680](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/680)).

### Systems

#### Added

* More comprehensive error messages and tracebacks in the Triton responses when errors occur inside DAG operators ([#343](https://github.com/NVIDIA-Merlin/systems/pull/343))

#### Major Changes

* The integration for the Feast feature store has been updated to be compatible with Feast 0.31, the latest release ([#344](https://github.com/NVIDIA-Merlin/systems/pull/344))

### Core

#### Added

* `Rename` operator, which was migrated from NVTabular and can now be used in all Merlin DAGs, like Systems ensembles and Dataloader transformations ([#312](https://github.com/NVIDIA-Merlin/core/pull/312))
* An `as_tensor_type` method on `TensorColumn` for converting column data across frameworks like NumPy, CuPy, Tensorflow, and Torch ([#285](https://github.com/NVIDIA-Merlin/core/pull/285))
* A `schema` parameter that allows converting a dataframe to a `TensorTable` in a way that produces fixed-length list columns with only values and not offsets ([#286](https://github.com/NVIDIA-Merlin/core/pull/285))

#### Major Changes

* The default DAG executors now automatically adjust the format of data passed between operators to ensure that it's in a format the receiving operator can process. This functionality was already present in parts of Merlin, but has been generalized to work in all Merlin DAGs. ([#280](https://github.com/NVIDIA-Merlin/core/pull/280))
* The list of mutually exclusive column tags has been expanded to include `Tags.EMBEDDING`, so that columns can't be mistakenly labeled as any combination of `ID`, `CONTINUOUS`, and `EMBEDDING` at the same time ([#316](https://github.com/NVIDIA-Merlin/core/pull/316))

#### Deprecated / Removed

* Numpy aliases for built-in Python types were deprecated in NumPy 1.20, and have been removed from Merlin Core ([#308](https://github.com/NVIDIA-Merlin/core/pull/308))

### Examples

#### Added

* Quick-start for Ranking - Added documentation, scripts and example notebook to export an inference pipeline for ranking models to Triton and send recommendation requests. The inference pipeline is a Triton ensemble which includes the NVTabular preprocessing workflow and the saved trained model ([#966](https://github.com/NVIDIA-Merlin/Merlin/pull/966))

#### Bugs

* Quick-start for Ranking  - Fixed error when saving model trained with single task.


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

* [Quick-start for Ranking](https://github.com/NVIDIA-Merlin/Merlin/blob/stable/examples/quick_start/ranking.md) - A new guide on how to preprocess data and, build, train, evaluate and hypertune Ranking models with Merlin, including best practices to improve models accuracy ([#915](https://github.com/NVIDIA-Merlin/Merlin/pull/915))
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
