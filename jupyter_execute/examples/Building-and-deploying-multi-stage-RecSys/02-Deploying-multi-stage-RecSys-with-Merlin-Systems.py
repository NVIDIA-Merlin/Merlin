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
# ================================


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# ## Deploying a Multi-Stage RecSys into Production with Merlin Systems and Triton Inference Server
# 
# This notebook is created using the latest stable [merlin-tensorflow-inference](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference/tags) container. 
# 
# At this point, when you reach out to this notebook, we expect that you have already executed the first notebook `01-Building-Recommender-Systems-with-Merlin.ipynb` and exported all the required files and models. 
# 
# We are going to generate recommended items for a given user query (user_id) by following the steps described in the figure below.

# ![tritonensemble](../images/triton_ensemble.png)

# Merlin Systems library have the set of operators to be able to serve multi-stage recommender systems built with Tensorflow on [Triton Inference Server](https://github.com/triton-inference-server/server)(TIS) easily and efficiently. Below, we will go through these operators and demonstrate their usage in serving a multi-stage system on Triton.

# ### Import required libraries and functions

# At this step, we assume you already installed the tensorflow-gpu (or -cpu), feast and faiss-gpu (or -cpu) libraries when running the first notebook `01-Building-Recommender-Systems-with-Merlin.ipynb`. 
# 
# In case you need to install them for running this example on GPU, execute the following script in a cell.
# ```
# %pip install tensorflow "feast<0.20" faiss-gpu
# ```
# or the following script in a cell for CPU.
# ```
# %pip install tensorflow-cpu "feast<0.20" faiss-cpu
# ```

# In[ ]:


import os
import numpy as np
import pandas as pd
import feast
import faiss
import seedir as sd
from nvtabular import ColumnSchema, Schema

from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.session_filter import FilterCandidates
from merlin.systems.dag.ops.softmax_sampling import SoftmaxSampling
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.unroll_features import UnrollFeatures
from merlin.systems.triton.utils import run_triton_server, run_ensemble_on_tritonserver


# ### Register our features on feature store

# The Feast feature registry is a central catalog of all the feature definitions and their related metadata(read more [here](https://docs.feast.dev/getting-started/architecture-and-components/registry)). We have defined our user and item features definitions in the `user_features.py` and  `item_features.py` files. With FeatureView() users can register data sources in their organizations into Feast, and then use those data sources for both training and online inference. In the `user_features.py` and `item_features.py` files, we are telling Feast where to find user and item features.
# 
# Before we move on to the next steps, we need to perform `feast apply`command as directed below.  With that, we register our features, we can apply the changes to create our feature registry and store all entity and feature view definitions in a local SQLite online store called `online_store.db`.

# In[3]:


BASE_DIR = os.environ.get("BASE_DIR", "/Merlin/examples/Building-and-deploying-multi-stage-RecSys/")

# define feature repo path
feast_repo_path = BASE_DIR + "feature_repo/"


# In[5]:


get_ipython().run_line_magic('cd', '$feast_repo_path')
get_ipython().system('feast apply')


# ### Loading features from offline store into an online store 
# 
# After we execute `apply` and registered our features and created our online local store, now we need to perform [materialization](https://docs.feast.dev/how-to-guides/running-feast-in-production) operation. This is done to keep our online store up to date and get it ready for prediction. For that we need to run a job that loads feature data from our feature view sources into our online store. As we add new features to our offline stores, we can continuously materialize them to keep our online store up to date by finding the latest feature values for each user. 
# 
# When you run the `feast materialize ..` command below, you will see a message <i>Materializing 2 feature views from 1995-01-01 01:01:01+00:00 to 2025-01-01 01:01:01+00:00 into the sqlite online store </i>  will be printed out.
# 
# Note that materialization step takes some time.. 

# In[6]:


get_ipython().system('feast materialize 1995-01-01T01:01:01 2025-01-01T01:01:01')


# Now, let's check our feature_repo structure again after we ran `apply` and `materialize` commands.

# In[4]:


# set up the base dir to for feature store
feature_repo_path = os.path.join(BASE_DIR, 'feature_repo')
sd.seedir(feature_repo_path, style='lines', itemlimit=10, depthlimit=5, exclude_folders=['.ipynb_checkpoints', '__pycache__'], sort=True)


# ### Set up Faiss index, create feature store client and objects for the Triton ensemble

# Create a folder for faiss index path

# In[8]:


if not os.path.isdir(os.path.join(BASE_DIR + 'faiss_index')):
    os.makedirs(os.path.join(BASE_DIR + 'faiss_index'))


# Define paths for ranking model, retrieval model, and faiss index path

# In[9]:


faiss_index_path = BASE_DIR + 'faiss_index' + "/index.faiss"
retrieval_model_path = BASE_DIR + "query_tower/"
ranking_model_path = BASE_DIR + "dlrm/"


# Create a request schema that we are going to use when sending a request to Triton Infrence Server (TIS).

# In[10]:


request_schema = Schema(
    [
        ColumnSchema("user_id", dtype=np.int32),
    ]
)


# `QueryFaiss` operator creates an interface between a FAISS Approximate Nearest Neighbors (ANN) Index and Triton Infrence Server. For a given input query vector, we do an ANN search query to find the ids of top-k nearby nodes in the index.
# 
# `setup_faiss` is  a utility function that will create a Faiss index from an embedding vector with using L2 distance.

# In[11]:


from merlin.systems.dag.ops.faiss import QueryFaiss, setup_faiss 

item_embeddings = np.ascontiguousarray(
    pd.read_parquet(BASE_DIR + "item_embeddings.parquet").to_numpy()
)
setup_faiss(item_embeddings, faiss_index_path)


# Create feature store client.

# In[12]:


feature_store = feast.FeatureStore(feast_repo_path)


# Fetch user features with `QueryFeast` operator from the feature store. `QueryFeast` operator is responsible for ensuring that our feast feature store can communicate correctly with tritonserver for the ensemble feast feature look ups.

# In[13]:


from merlin.systems.dag.ops.feast import QueryFeast 

user_features = ["user_id"] >> QueryFeast.from_feature_view(
    store=feature_store,
    view="user_features",
    column="user_id",
    include_id=True,
)


# Retrieve top-K candidate items using `retrieval model` that are relevant for a given user. We use `PredictTensorflow()` operator that takes a tensorflow model and packages it correctly for TIS to run with the tensorflow backend.

# In[14]:


# prevent TF to claim all GPU memory
from merlin.models.loader.tf_utils import configure_tensorflow

configure_tensorflow()


# In[15]:


topk_retrieval = 100
retrieval = (
    user_features
    >> PredictTensorflow(retrieval_model_path)
    >> QueryFaiss(faiss_index_path, topk=topk_retrieval)
)


# Fetch item features for the candidate items that are retrieved from the retrieval step above from the feature store.

# In[16]:


item_features = retrieval["candidate_ids"] >> QueryFeast.from_feature_view(
    store=feature_store,
    view="item_features",
    column="candidate_ids",
    output_prefix="item",
    include_id=True,
)


# Merge the user features and items features to create the all set of combined features that were used in model training using `UnrollFeatures` operator which takes a target column and joins the "unroll" columns to the target. This helps when broadcasting a series of user features to a set of items.

# In[17]:


user_features_to_unroll = [
    "user_id",
    "user_shops",
    "user_profile",
    "user_group",
    "user_gender",
    "user_age",
    "user_consumption_2",
    "user_is_occupied",
    "user_geography",
    "user_intentions",
    "user_brands",
    "user_categories",
]

combined_features = item_features >> UnrollFeatures(
    "item_id", user_features[user_features_to_unroll]
)


# Rank the combined features using the trained ranking model, which is a DLRM model for this example. We feed the path of the ranking model to `PredictTensorflow()` operator.

# In[18]:


ranking = combined_features >> PredictTensorflow(ranking_model_path)


# For the ordering we use `SoftmaxSampling()` operator. This operator sorts all inputs in descending order given the input ids and prediction introducing some randomization into the ordering by sampling items from the softmax of the predicted relevance scores, and finally returns top-k ordered items.

# In[19]:


top_k=10
ordering = combined_features["item_id"] >> SoftmaxSampling(
    relevance_col=ranking["output_1"], topk=top_k, temperature=20.0
)


# ### Export Graph as Ensemble
# The last step is to create the ensemble artifacts that TIS can consume. To make these artifacts import the Ensemble class. This class  represents an entire ensemble consisting of multiple models that run sequentially in TIS initiated by an inference request. It is responsible with interpreting the graph and exporting the correct files for TIS.
# 
# When we create an Ensemble object we feed the graph and a schema representing the starting input of the graph.  After we create the ensemble object, we export the graph, supplying an export path for the `ensemble.export()` function. This returns an ensemble config which represents the entire inference pipeline and a list of node-specific configs.

# Create the folder to export the models and config files.

# In[20]:


if not os.path.isdir(os.path.join(BASE_DIR + 'poc_ensemble')):
    os.makedirs(os.path.join(BASE_DIR + 'poc_ensemble'))


# In[21]:


# define the path where all the models and config files exported to
export_path = os.path.join(BASE_DIR + 'poc_ensemble')

ensemble = Ensemble(ordering, request_schema)
ens_config, node_configs = ensemble.export(export_path)


# Let's check our export_path structure

# In[32]:


sd.seedir(export_path, style='lines', itemlimit=10, depthlimit=5, exclude_folders=['.ipynb_checkpoints', '__pycache__'], sort=True)


# ### Starting Triton Server

# It is time to deploy all the models as an ensemble model to Triton Inference Serve [TIS](https://github.com/triton-inference-server). After we export the ensemble, we are ready to start the TIS. You can start triton server by using the following command on your terminal:
# 
# ```
# tritonserver --model-repository=/ensemble_export_path/ --backend-config=tensorflow,version=2
# ```
# 
# For the `--model-repository` argument, specify the same path as the `export_path` that you specified previously in the `ensemble.export` method. This command will launch the server and load all the models to the server. Once all the models are loaded successfully, you should see `READY` status printed out in the terminal for each loaded model.

# ### Retrieving Recommendations from Triton

# Once our models are successfully loaded to the TIS, we can now easily send a request to TIS and get a response for our query with `send_triton_request` utility function.

# In[23]:


from merlin.systems.triton.utils import send_triton_request
from merlin.core.dispatch import make_df

# create a request to be sent to TIS
request = make_df({"user_id": [1]})
request["user_id"] = request["user_id"].astype(np.int32)

outputs = ensemble.graph.output_schema.column_names


# In[26]:


response = send_triton_request(request, outputs)
response


# Note that these item ids are encoded values, not the raw original values. We will eventually create the reverse dictionary lookup functionality to be able to map these encoded item ids to their original raw ids with one-line of code. But if you really want to do it now, you can easily map these ids to their original values using the `unique.item_id.parquet` file stored in the `categories` folder.
# 
# That's it! You finished deploying a multi-stage Recommender Systems on Triton Inference Server using Merlin framework.
