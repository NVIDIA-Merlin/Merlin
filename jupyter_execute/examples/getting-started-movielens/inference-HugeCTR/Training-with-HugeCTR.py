#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# ## Overview
# 
# In this notebook, we want to provide an overview what HugeCTR framework is, its features and benefits. We will use HugeCTR to train a basic neural network architecture and deploy the saved model to Triton Inference Server. 
# 
# <b>Learning Objectives</b>:
# * Adopt NVTabular workflow to provide input files to HugeCTR
# * Define HugeCTR neural network architecture
# * Train a deep learning model with HugeCTR
# * Deploy HugeCTR to Triton Inference Server

# ### Why using HugeCTR?
# 
# HugeCTR is a GPU-accelerated recommender framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs).<br>
# 
# HugeCTR offers multiple advantages to train deep learning recommender systems:
# 1. **Speed**: HugeCTR is a highly efficient framework written C++. We experienced up to 10x speed up. HugeCTR on a NVIDIA DGX A100 system proved to be the fastest commercially available solution for training the architecture Deep Learning Recommender Model (DLRM) developed by Facebook.
# 2. **Scale**: HugeCTR supports model parallel scaling. It distributes the large embedding tables over multiple GPUs or multiple nodes. 
# 3. **Easy-to-use**: Easy-to-use Python API similar to Keras. Examples for popular deep learning recommender systems architectures (Wide&Deep, DLRM, DCN, DeepFM) are available.

# ### Other Features of HugeCTR
# 
# HugeCTR is designed to scale deep learning models for recommender systems. It provides a list of other important features:
# * Proficiency in oversubscribing models to train embedding tables with single nodes that don’t fit within the GPU or CPU memory (only required embeddings are prefetched from a parameter server per batch)
# * Asynchronous and multithreaded data pipelines
# * A highly optimized data loader.
# * Supported data formats such as parquet and binary
# * Integration with Triton Inference Server for deployment to production
# 

# ### Getting Started

# In this example, we will train a neural network with HugeCTR. We will use NVTabular for preprocessing.

# #### Preprocessing and Feature Engineering with NVTabular
# 
# We use NVTabular to `Categorify` our categorical input columns.

# In[2]:


# External dependencies
import os
import shutil
import gc

import nvtabular as nvt
import cudf
import numpy as np

from os import path
from sklearn.model_selection import train_test_split

from nvtabular.utils import download_file


# We define our base directory, containing the data.

# In[3]:


# path to store raw and preprocessed data
BASE_DIR = "/model/data/"


# If the data is not available in the base directory, we will download and unzip the data.

# In[4]:


download_file(
    "http://files.grouplens.org/datasets/movielens/ml-25m.zip", os.path.join(BASE_DIR, "ml-25m.zip")
)


# ## Preparing the dataset with NVTabular

# First, we take a look at the movie metadata.
# 
# Let's load the movie ratings.

# In[5]:


ratings = cudf.read_csv(os.path.join(BASE_DIR, "ml-25m", "ratings.csv"))
ratings.head()


# We drop the timestamp column and split the ratings into training and test dataset. We use a simple random split.

# In[6]:


ratings = ratings.drop("timestamp", axis=1)
train, valid = train_test_split(ratings, test_size=0.2, random_state=42)


# In[7]:


train.head()


# We save our train and valid datasets as parquet files on disk, and below we will read them in while initializing the Dataset objects.

# In[8]:


train.to_parquet(BASE_DIR + "train.parquet")
valid.to_parquet(BASE_DIR + "valid.parquet")


# In[9]:


del train
del valid
gc.collect()


# Let's define our categorical and label columns. Note that in that example we do not have numerical columns.

# In[10]:


CATEGORICAL_COLUMNS = ["userId", "movieId"]
LABEL_COLUMNS = ["rating"]


# Let's add Categorify op for our categorical features, userId, movieId.

# In[11]:


cat_features = CATEGORICAL_COLUMNS >> nvt.ops.Categorify(cat_cache="device")


# The ratings are on a scale between 1-5. We want to predict a binary target with 1 are all ratings >=4 and 0 are all ratings <=3. We use the LambdaOp for it.

# In[12]:


ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))


# We can visualize our calculation graph.

# In[13]:


output = cat_features + ratings
(output).graph


# We initialize our NVTabular workflow.

# In[14]:


workflow = nvt.Workflow(output)


# We initialize NVTabular Datasets, and use the part_size parameter, which defines the size read into GPU-memory at once, in nvt.Dataset.

# In[15]:


train_dataset = nvt.Dataset(BASE_DIR + "train.parquet", part_size="100MB")
valid_dataset = nvt.Dataset(BASE_DIR + "valid.parquet", part_size="100MB")


# First, we collect the training dataset statistics.

# In[16]:


get_ipython().run_cell_magic('time', '', '\nworkflow.fit(train_dataset)\n')


# This step is slightly different for HugeCTR. HugeCTR expect the categorical input columns as `int64` and continuous/label columns as `float32`  We can define output datatypes for our NVTabular workflow.

# In[17]:


dict_dtypes = {}

for col in CATEGORICAL_COLUMNS:
    dict_dtypes[col] = np.int64

for col in LABEL_COLUMNS:
    dict_dtypes[col] = np.float32


# Note: We do not have numerical output columns

# In[18]:


train_dir = os.path.join(BASE_DIR, "train")
valid_dir = os.path.join(BASE_DIR, "valid")
if path.exists(train_dir):
    shutil.rmtree(train_dir)
if path.exists(valid_dir):
    shutil.rmtree(valid_dir)


# In addition, we need to provide the data schema to the output calls. We need to define which output columns are `categorical`, `continuous` and which is the `label` columns. NVTabular will write metadata files, which HugeCTR requires to load the data and optimize training.

# In[19]:


workflow.transform(train_dataset).to_parquet(
    output_path=BASE_DIR + "train/",
    shuffle=nvt.io.Shuffle.PER_PARTITION,
    cats=CATEGORICAL_COLUMNS,
    labels=LABEL_COLUMNS,
    dtypes=dict_dtypes,
)
workflow.transform(valid_dataset).to_parquet(
    output_path=BASE_DIR + "valid/",
    shuffle=False,
    cats=CATEGORICAL_COLUMNS,
    labels=LABEL_COLUMNS,
    dtypes=dict_dtypes,
)


# ## Scaling Accelerated training with HugeCTR

# HugeCTR is a deep learning framework dedicated to recommendation systems. It is written in CUDA C++. As HugeCTR optimizes the training in CUDA++, we need to define the training pipeline and model architecture and execute it via the commandline. We will use the Python API, which is similar to Keras models.

# HugeCTR has three main components:
# * Solver: Specifies various details such as active GPU list, batchsize, and model_file
# * Optimizer: Specifies the type of optimizer and its hyperparameters
# * Model: Specifies training/evaluation data (and their paths), embeddings, and dense layers. Note that embeddings must precede the dense layers

# **Solver**
# 
# Let's take a look on the parameter for the `Solver`. We should be familiar from other frameworks for the hyperparameter.
# 
# ```
# solver = hugectr.solver_parser_helper(
# - vvgpu: GPU indices used in the training process, which has two levels. For example: [[0,1],[1,2]] indicates that two nodes are used in the first node. GPUs 0 and 1 are used while GPUs 1 and 2 are used for the second node. It is also possible to specify non-continuous GPU indices such as [0, 2, 4, 7]  
# - max_iter: Total number of training iterations
# - batchsize: Minibatch size used in training
# - display: Intervals to print loss on the screen
# - eval_interval: Evaluation interval in the unit of training iteration
# - max_eval_batches: Maximum number of batches used in evaluation. It is recommended that the number is equal to or bigger than the actual number of bathces in the evaluation dataset.
# If max_iter is used, the evaluation happens for max_eval_batches by repeating the evaluation dataset infinitely.
# On the other hand, with num_epochs, HugeCTR stops the evaluation if all the evaluation data is consumed    
# - batchsize_eval: Maximum number of batches used in evaluation. It is recommended that the number is equal to or
#   bigger than the actual number of bathces in the evaluation dataset
# - mixed_precision: Enables mixed precision training with the scaler specified here. Only 128,256, 512, and 1024 scalers are supported
# )
# ```

# **Optimizer**
# 
# The optimizer is the algorithm to update the model parameters. HugeCTR supports the common algorithms.
# 
# 
# ```
# optimizer = CreateOptimizer(
# - optimizer_type: Optimizer algorithm - Adam, MomentumSGD, Nesterov, and SGD 
# - learning_rate: Learning Rate for optimizer
# )
# ```

# **Model**
# 
# We initialize the model with the solver and optimizer:
# 
# ```
# model = hugectr.Model(solver, optimizer)
# ```
# 
# We can add multiple layers to the model with `model.add` function. We will focus on:
# - `Input` defines the input data
# - `SparseEmbedding` defines the embedding layer
# - `DenseLayer` defines dense layers, such as fully connected, ReLU, BatchNorm, etc.
# 
# **HugeCTR organizes the layers by names. For each layer, we define the input and output names.**

# Input layer:
# 
# This layer is required to define the input data.
# 
# ```
# hugectr.Input(
#     data_reader_type: Data format to read
#     source: The training dataset file list.
#     eval_source: The evaluation dataset file list.
#     check_type: The data error detection machanism (Sum: Checksum, None: no detection).
#     label_dim: Number of label columns
#     label_name: Name of label columns in network architecture
#     dense_dim: Number of continous columns
#     dense_name: Name of contiunous columns in network architecture
#     slot_size_array: The list of categorical feature cardinalities
#     data_reader_sparse_param_array: Configuration how to read sparse data
#     sparse_names: Name of sparse/categorical columns in network architecture
# )
# ```
# 
# SparseEmbedding:
# 
# This layer defines embedding table
# 
# ```
# hugectr.SparseEmbedding(
#     embedding_type: Different embedding options to distribute embedding tables 
#     max_vocabulary_size_per_gpu: Maximum vocabulary size or cardinality across all the input features
#     embedding_vec_size: Embedding vector size
#     combiner: Intra-slot reduction op (0=sum, 1=average)
#     sparse_embedding_name: Layer name
#     bottom_name: Input layer names
# )
# ```
# 
# DenseLayer:
# 
# This layer is copied to each GPU and is normally used for the MLP tower.
# 
# ```
# hugectr.DenseLayer(
#     layer_type: Layer type, such as FullyConnected, Reshape, Concat, Loss, BatchNorm, etc.
#     bottom_names: Input layer names
#     top_names: Layer name
#     ...: Depending on the layer type additional parameter can be defined
# )
# ```

# ## Let's define our model
# 
# We walked through the documentation, but it is useful to understand the API. Finally, we can define our model. We will write the model to `./model.py` and execute it afterwards.

# We need the cardinalities of each categorical feature to assign as `slot_size_array` in the model below.

# In[20]:


from nvtabular.ops import get_embedding_sizes

embeddings = get_embedding_sizes(workflow)
print(embeddings)


# In addition, we need the total cardinalities to be assigned as `max_vocabulary_size_per_gpu` parameter.

# In[21]:


total_cardinality = embeddings["movieId"][0] + embeddings["userId"][0]
total_cardinality


# In[22]:


get_ipython().run_cell_magic('writefile', "'./model.py'", '\nimport hugectr\nfrom mpi4py import MPI  # noqa\n\nsolver = hugectr.solver_parser_helper(\n    vvgpu=[[0]],\n    max_iter=2000,\n    batchsize=2048,\n    display=100,\n    eval_interval=200,\n    batchsize_eval=2048,\n    max_eval_batches=160,\n    i64_input_key=True,\n    use_mixed_precision=False,\n    repeat_dataset=True,\n    snapshot=1900,\n)\noptimizer = hugectr.optimizer.CreateOptimizer(\n    optimizer_type=hugectr.Optimizer_t.Adam, use_mixed_precision=False\n)\nmodel = hugectr.Model(solver, optimizer)\n\nmodel.add(\n    hugectr.Input(\n        data_reader_type=hugectr.DataReaderType_t.Parquet,\n        source="/model/data/train/_file_list.txt",\n        eval_source="/model/data/valid/_file_list.txt",\n        check_type=hugectr.Check_t.Non,\n        label_dim=1,\n        label_name="label",\n        dense_dim=0,\n        dense_name="dense",\n        slot_size_array=[56586, 162542],\n        data_reader_sparse_param_array=[\n            hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 3, 1, 2)\n        ],\n        sparse_names=["data1"],\n    )\n)\nmodel.add(\n    hugectr.SparseEmbedding(\n        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,\n        max_vocabulary_size_per_gpu=219128,\n        embedding_vec_size=16,\n        combiner=0,\n        sparse_embedding_name="sparse_embedding1",\n        bottom_name="data1",\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.Reshape,\n        bottom_names=["sparse_embedding1"],\n        top_names=["reshape1"],\n        leading_dim=32,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["reshape1"],\n        top_names=["fc1"],\n        num_output=128,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU,\n        bottom_names=["fc1"],\n        top_names=["relu1"],\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu1"],\n        top_names=["fc2"],\n        num_output=128,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU,\n        bottom_names=["fc2"],\n        top_names=["relu2"],\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu2"],\n        top_names=["fc3"],\n        num_output=1,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,\n        bottom_names=["fc3", "label"],\n        top_names=["loss"],\n    )\n)\nmodel.compile()\nmodel.summary()\nmodel.fit()\n')


# In[23]:


get_ipython().system('python model.py')


# We trained our model.

# After training terminates, we can see that two `.model` files are generated. We need to move them inside `1` folder under the `movielens_hugectr` folder. Let's create these folders first.

# In[24]:


get_ipython().system('mkdir -p /model/movielens_hugectr/1')


# Now we move our saved `.model` files inside `1` folder.

# In[25]:


get_ipython().system('mv *.model /model/movielens_hugectr/1/')


# Note that these stored `.model` files will be used in the inference. Now we have to create a JSON file for inference which has a similar configuration as our training file. We should remove the solver and optimizer clauses and add the inference clause in the JSON file. The paths of the stored dense model and sparse model(s) should be specified at dense_model_file and sparse_model_file within the inference clause. We need to make some modifications to data in the layers clause. Besides, we need to change the last layer from BinaryCrossEntropyLoss to Sigmoid. The rest of "layers" should be exactly the same as that in the training model.py file.
# 
# Now let's create a `movielens.json` file inside the `movielens/1` folder. We have already retrieved the cardinality of each categorical column using `get_embedding_sizes` function above. We will use these cardinalities below in the `movielens.json` file as well.

# In[26]:


get_ipython().run_cell_magic('writefile', "'/model/movielens_hugectr/1/movielens.json'", '\n{\n    "inference": {\n        "max_batchsize": 64,\n        "hit_rate_threshold": 0.6,\n        "dense_model_file": "/model/models/movielens/1/_dense_1900.model",\n        "sparse_model_file": "/model/models/movielens/1/0_sparse_1900.model",\n        "label": 1,\n        "input_key_type": "I64"\n    },\n    "layers": [\n        {\n            "name": "data",\n            "type": "Data",\n            "format": "Parquet",\n            "slot_size_array": [56586, 162542],\n            "source": "/model/data/train/_file_list.txt",\n            "eval_source": "/model/data/valid/_file_list.txt",\n            "check": "Sum",\n            "label": {"top": "label", "label_dim": 1},\n            "dense": {"top": "dense", "dense_dim": 0},\n            "sparse": [\n                {\n                    "top": "data1",\n                    "type": "DistributedSlot",\n                    "max_feature_num_per_sample": 3,\n                    "slot_num": 2\n                }\n            ]\n        },\n        {\n            "name": "sparse_embedding1",\n            "type": "DistributedSlotSparseEmbeddingHash",\n            "bottom": "data1",\n            "top": "sparse_embedding1",\n            "sparse_embedding_hparam": {\n                "max_vocabulary_size_per_gpu": 219128,\n                "embedding_vec_size": 16,\n                "combiner": 0\n            }\n        },\n        {\n            "name": "reshape1",\n            "type": "Reshape",\n            "bottom": "sparse_embedding1",\n            "top": "reshape1",\n            "leading_dim": 32\n        },\n        {\n            "name": "fc1",\n            "type": "InnerProduct",\n            "bottom": "reshape1",\n            "top": "fc1",\n            "fc_param": {"num_output": 128}\n        },\n        {"name": "relu1", "type": "ReLU", "bottom": "fc1", "top": "relu1"},\n        {\n            "name": "fc2",\n            "type": "InnerProduct",\n            "bottom": "relu1",\n            "top": "fc2",\n            "fc_param": {"num_output": 128}\n        },\n        {"name": "relu2", "type": "ReLU", "bottom": "fc2", "top": "relu2"},\n        {\n            "name": "fc3",\n            "type": "InnerProduct",\n            "bottom": "relu2",\n            "top": "fc3",\n            "fc_param": {"num_output": 1}\n        },\n        {"name": "sigmoid", "type": "Sigmoid", "bottom": "fc3", "top": "sigmoid"}\n    ]\n}\n')


# Now we can save our models to be deployed at the inference stage. To do so we will use `export_hugectr_ensemble` method below. With this method, we can generate the `config.pbtxt` files automatically for each model. In doing so, we should also create a `hugectr_params` dictionary, and define the parameters  like where the `movielens.json` file will be read, `slots` which corresponds to number of categorical features, `embedding_vector_size`, `max_nnz`, and `n_outputs` which is number of outputs.

# The script below creates an ensemble triton server model where  
# 
# - `workflow` is the the nvtabular workflow used in preprocessing, 
# - `hugectr_model_path` is the HugeCTR model that should be served. This path includes the `.model` files.
# - `name` is the base name of the various triton models
# - `output_path` is the path where is model will be saved to.

# In[27]:


from nvtabular.inference.triton import export_hugectr_ensemble

hugectr_params = dict()
hugectr_params["config"] = "/model/models/movielens/1/movielens.json"
hugectr_params["slots"] = 2
hugectr_params["max_nnz"] = 2
hugectr_params["embedding_vector_size"] = 16
hugectr_params["n_outputs"] = 1
export_hugectr_ensemble(
    workflow=workflow,
    hugectr_model_path="/model/movielens_hugectr/1/",
    hugectr_params=hugectr_params,
    name="movielens",
    output_path="/model/models/",
    label_columns=["rating"],
    cats=CATEGORICAL_COLUMNS,
    max_batch_size=64,
)


# After we run the script above, we will have three model folders saved as `movielens_nvt`, `movielens` and `movielens_ens`. Now we can move to the next notebook, `movielens-HugeCTR-inference`, to send request to the Triton Inference Server using the saved ensemble model.