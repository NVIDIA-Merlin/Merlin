# About Merlin Operators

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## Understanding Operators

Merlin uses Operators to perform computation on datasets such as normalizing continuous variables, bucketing continuous variables, clipping variables between minimum and maximum values, and so on.

An Operator implements two key methods:

Fit
: The `fit` method performs any pre-computation steps that are required before modifying the data.

  For example, the `Normalize` Operator normalizes the values of a continuous variable between `0` and `1`.
  The `fit` method determines the minimum and maximum values.

  The method is optional.
  For example, the `Bucketize` and `Clip` Operators do not implement the method because you specify the bucket boundaries or the minimum and maximum values for clipping.
  These Operators do not need to access the data to perform any pre-computation steps.

Transform
: The `transform` method operates on the dataset such as normalizing values, bucketing, or clipping.
  This method modifies the data.

Another difference between the two methods is that the `fit` method accepts a Merlin dataset object and the `transform` method accepts a DataFrame object.
The difference is an implementation detail---the `fit` method must access all the data and the `transform` method processes each part of the dataset one at a time.

```{code-block} python
---
emphasize-lines: 5, 12
---
# Typical signature of a fit method.
def fit(
    self,
    selector: ColumnSelector,
    dataset: Dataset
) -> Any

# Typical signature of a transform method.
def transform(
    self,
    selector: ColumnSelector,
    df: DataFrame
) -> DataFrame
```

## Operators and Columns: Column Selector

In most cases, you want an Operator to process a subset of the columns in your input dataset.
Both the `fit` and `transform` methods have a `selector` argument that specifies the columns to operate on.
Merlin uses a `ColumnSelector` class to represent the columns.

The simplest column selector is a list of strings that specify some column names.
In the following sample code, `["col1", "col2"]` become an instance of a `ColumnSelector` class.

```python
result = ["col1", "col2"] >> SomeOperator(...)
```

Column selectors also offer a more powerful and flexible way to specify columns.
You can specify the input columns to an Operator with tags.
In the following sample code, the Operator processes all the continuous variables in a dataset.

```python
result = [Tags.CONTINUOUS] >> SomeOperator(...)
```

Using tags to create a column selector offers the following advantages:

- Enables you to apply several Operators to the same kind of columns, such as categorical or continuous variables.
- Reduces code maintenance by enabling your code to automatically operate on newly added columns in a dataset.
- Simplifies code by avoiding lists of strings for column names.

## How to Build an Operator

Blah.

## Reference Documentation

- {py:class}`merlin.dag.BaseOperator`
- {py:class}`merlin.dag.ColumnSelector`
- {py:class}`merlin.schema.Tags`
- {py:class}`merlin.io.DataSet`