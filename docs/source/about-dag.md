# About the Merlin Graph

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## Purpose of the Merlin Graph

Merlin uses a directed acyclic graph (DAG) to represent operations on data such as normalizing or clipping values and to represent operations in a recommender system such as creating an ensemble or filtering candidate items during inference.

Understanding the Merlin DAG is helpful if you want to develop your own Operator or building a recommender system with Merlin.

## Graph Terminology

node
: A node in the DAG is a group of columns and at least one _operator_.
  The columns are specified with a _column selector_.
  A node has an _input schema_ and an _output schema_.
  Resolution of the schemas is delayed until you run `fit` or `transform` on a dataset.

column selector
: A column selector specifies the columns to select from a dataset using column names or _tags_.

operator
: An operator performs a transformation on data and return a new _node_.
  The data is identified by the _column selector_.
  Some simple operators like `+` and `-` add or remove columns.
  More complex operations are applied by shifting the operators onto the column selector with the `>>` notation.

schema
: A Merlin schema is metadata that describes the columns in a dataset.
  Each column has its own schema that identifies the column name and can specify _tags_ and properties.

tag
: A Merlin tag categorizes information about a column.
  Adding a tag to a column enables you to select columns for operations by tag rather than name.

  For example, you can add the `CONTINUOUS` or `CATEGORICAL` tags to columns.
  Feature engineering Operators, modeling, and inference operations can use that information to operate accordingly on the dataset.

## Introduction to Operators, Columns, Nodes, and Schema

The NVTabular library uses Operators for feature engineering.
One example of an NVTabular Operator is `Normalize`.
The Operator normalizes continuous variables between `0` and `1`.

The Merlin Systems library uses Operators for building ensembles and performing inference.
The library includes Operators such as `FilterCandidates` and `PredictTensorflow`.
You use these Operators to put your models into production and serve recommendations.

Merlin enables you to chain together Operators with the `>>` syntax to create feature-processing workflows.
The `>>` syntax means "take the output columns from the left-hand side and feed them as the input columns to the right-hand side."

You can specify an explicit list of columns names for an Operator.
The following code block shows the syntax for explicit column names:

```python
result = ["col1", "col2",] >> SomeOperator(...)
```

Or, you can use the `>>` syntax between Operators to run one Operator on all the output columns from the preceding Operator:

```python
result = AnOperator(...) >> OtherOperator(...)
```

Chaining Operators together builds a graph.
The following figure shows how each node in the graph has an Operator.

![A directed graph with two nodes. The first node is a Selection Operator and selects columns "col1" and "col2." The second node receives the two columns as its input. The second node has a fictional SomeOperator Operator.](../images/graph_simple.svg)

```{tip}
After you build an NVTabular workflow or Merlin Systems transform workflow, you can visualize the graph and create an image like the preceding example by running the `graph` method.
```

Each node in a graph has an input schema and an output schema that describe the input columns to the Operator and the output columns produced by the Operator.
The following figure represents an Operator, `SomeOperator`, that adds `colB` to a dataset.

![Part of a directed graph that shows the input schema to a fictional SomeOperator Operator as "colA". The fictional Operator adds adds "colB" and the result is an output schema with "colA" and "colB."](../images/graph_schema.svg)

In practice, when Merlin first builds the graph, the workflow does not initially know which columns are input or output.
This is for two reasons:

1. Merlin enables you to build graphs that process categories of columns.
   The categories are specified by _tags_ instead of an explicit list of column names.

   For example, you can select the continuous columns from your dataset with code like the following example:

   ```python
   [Tags.CONTINUOUS] >> Operator(...)
   ```

1. You can chain Operators together into a graph, such as an NVTabular workflow, before you specify a dataset.
   The graph, Operators, and schema do not know which columns will be selected by tag until the software accesses the dataset and determines the column names.

## Reference Documentation

- {py:class}`nvtabular.ops.Normalize`
- {py:class}`nvtabular.workflow.workflow.Workflow`
- {py:class}`merlin.systems.dag.ops.workflow.TransformWorkflow`
- {py:class}`merlin.systems.dag.Ensemble`
- {py:class}`merlin.systems.dag.ops.session_filter.FilterCandidates`
- {py:class}`merlin.systems.dag.tensorflow.PredictTensorFlow`