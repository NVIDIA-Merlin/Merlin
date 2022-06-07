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
# ## Building Intelligent Recommender Systems with Merlin
# 
# This notebook is created using the latest stable [merlin-tensorflow-inference](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference/tags) container. 
# 
# ### Overview

# Recommender Systems (RecSys) are the engine of the modern internet and the catalyst for human decisions. Building a recommendation system is challenging because it requires multiple stages (data preprocessing, offline training, item retrieval, filtering, ranking, ordering, etc.) to work together seamlessly and efficiently. The biggest challenges for new practitioners are the lack of understanding around what RecSys look like in the real world, and the gap between examples of simple models and a production-ready end-to-end recommender systems.

# The figure below represents a four-stage recommender systems. This is more complex process than only training a single model and deploying it, and it is much more realistic and closer to what's happening in the real-world recommender production systems.

# ![fourstage](../images/fourstages.png)

# In these series of notebooks, we are going to showcase how we can deploy a four-stage recommender systems using Merlin Systems library easily on [Triton Inference Server](https://github.com/triton-inference-server/server). Let's go over the concepts in the figure briefly. 
# - **Retrieval:** This is the step to narrow down millions of items into thousands of candidates. We are going to train a Two-Tower item retrieval model to retrieve the relevant top-K candidate items.
# - **Filtering:** This step is to exclude the already interacted  or undesirable items from the candidate items set or to apply business logic rules. Although this is an important step, for this example we skip this step.
# - **Scoring:** This is also known as ranking. Here the retrieved and filtered candidate items are being scored. We are going to train a ranking model to be able to use at our scoring step. 
# - **Ordering:** At this stage, we can order the final set of items that we want to recommend to the user. Here, we’re able to align the output of the model with business needs, constraints, or criteria.
# 
# To learn more about the four-stage recommender systems, you can listen to Even Oldridge's [Moving Beyond Recommender Models talk](https://www.youtube.com/watch?v=5qjiY-kLwFY&list=PL65MqKWg6XcrdN4TJV0K1PdLhF_Uq-b43&index=7) at KDD'21 and read more [in this blog post](https://eugeneyan.com/writing/system-design-for-discovery/).

# ### Learning objectives
# - Understanding four stages of recommender systems
# - Training retrieval and ranking models with Merlin Models
# - Setting up feature store and approximate nearest neighbours (ANN) search libraries
# - Deploying trained models to Triton Inference Server with Merlin Systems

# In addition to NVIDIA Merlin libraries and the Triton Inference Server client library, we use two external libraries in these series of examples:
# 
# - [Feast](https://docs.feast.dev/): an end-to-end open source feature store library for machine learning
# - [Faiss](https://github.com/facebookresearch/faiss): a library for efficient similarity search and clustering of dense vectors
# 
# You can find more information about `Feast feature store` and `Faiss` libraries in the next notebook.

# ### Import required libraries and functions

# **Compatibility:**
# 
# These notebooks are developed and tested using our latest inference container on [NVIDIA's docker registry](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=dateModifiedDESC&query=merlin).

# In[ ]:


# for running this example on GPU, install the following libraries
get_ipython().run_line_magic('pip', 'install tensorflow "feast<0.20" faiss-gpu')

# for running this example on CPU, uncomment the following lines
# %pip install tensorflow-cpu "feast<0.20" faiss-cpu
# %pip uninstall cudf


# In[3]:


import os
import glob
import gc

import nvtabular as nvt
from nvtabular.ops import *

from merlin.models.utils.example_utils import workflow_fit_transform

from merlin.schema.tags import Tags

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
import tensorflow as tf

# for running this example on CPU, comment out the line below
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async" 


# In[4]:


# disable INFO and DEBUG logging everywhere
import logging
logging.disable(logging.WARNING)


# In this example notebook, we will generate the synthetic train and test datasets mimicking the real [Ali-CCP: Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408#1) dataset to build our recommender system models.
# 
# First, we define our input and output paths.

# In[5]:


DATA_FOLDER = os.environ.get("DATA_FOLDER", "/workspace/data/")
output_path = os.path.join(DATA_FOLDER, 'processed/ranking')


# Then, we use `generate_data` utility function to generate synthetic dataset. 

# In[6]:


from merlin.datasets.synthetic import generate_data

NUM_ROWS = 100000
train, valid = generate_data("aliccp-raw", int(NUM_ROWS), set_sizes=(0.7, 0.3))


# If you would like to use the real ALI-CCP dataset, you can use [get_aliccp()](https://github.com/NVIDIA-Merlin/models/blob/main/merlin/datasets/ecommerce/aliccp/dataset.py) function instead. This function takes the raw csv files, and generate parquet files that can be directly fed to NVTabular workflow above.

# ### Feature Engineering with NVTabular

# We are going to process our raw categorical features by encoding them using `Categorify()` operator and tag the features with `user` or `item` tags in the schema file. To learn more about [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) and the schema object visit this example [notebook](https://github.com/NVIDIA-Merlin/models/blob/main/examples/02-Merlin-Models-and-NVTabular-integration.ipynb) in the Merlin Models repo.

# In[7]:


get_ipython().run_cell_magic('time', '', '\nuser_id = ["user_id"] >> Categorify(dtype=\'int32\') >> TagAsUserID()\nitem_id = ["item_id"] >> Categorify(dtype=\'int32\') >> TagAsItemID()\n\nitem_features = ["item_category", "item_shop", "item_brand"] >> Categorify(dtype=\'int32\') >> TagAsItemFeatures() \n\nuser_features = [\'user_shops\', \'user_profile\', \'user_group\', \n       \'user_gender\', \'user_age\', \'user_consumption_2\', \'user_is_occupied\',\n       \'user_geography\', \'user_intentions\', \'user_brands\', \'user_categories\'] \\\n    >> Categorify(dtype=\'int32\') >> TagAsUserFeatures() \n\ntargets = ["click"] >> AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, "target"])\n\noutputs = user_id+item_id+item_features+user_features+targets\n')


# Let's call `transform_aliccp` utility function to be able to perform `fit` and `transform` steps on the raw dataset applying the operators defined in the NVTabular workflow pipeline below, and also save our workflow model. After fit and transform, the processed parquet files are saved to output_path.

# In[8]:


from merlin.datasets.ecommerce import transform_aliccp

transform_aliccp((train, valid), output_path, nvt_workflow=outputs, workflow_name='workflow_ranking')


# ### Training a Ranking Model with DLRM

# NVTabular exported the schema file, `schema.pbtxt` a protobuf text file, of our processed dataset. To learn more about the schema object and schema file you can explore [02-Merlin-Models-and-NVTabular-integration.ipynb](https://github.com/NVIDIA-Merlin/models/blob/main/examples/02-Merlin-Models-and-NVTabular-integration.ipynb) notebook.
# 
# We use the `schema` object to define our model.

# In[9]:


# define train and valid dataset objects
train = Dataset(os.path.join(output_path, 'train', '*.parquet'), part_size="500MB")
valid = Dataset(os.path.join(output_path, 'valid', '*.parquet'), part_size="500MB")

# define schema object
schema = train.schema


# In[10]:


target_column = schema.select_by_tag(Tags.TARGET).column_names[0]
target_column


# Deep Learning Recommendation Model [(DLRM)](https://arxiv.org/abs/1906.00091) architecture is a popular neural network model originally proposed by Facebook in 2019. The model was introduced as a personalization deep learning model that uses embeddings to process sparse features that represent categorical data and a multilayer perceptron (MLP) to process dense features, then interacts these features explicitly using the statistical techniques proposed in [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5694074). To learn more about DLRM architetcture please visit `Exploring-different-models` [notebook](https://github.com/NVIDIA-Merlin/models/blob/main/examples/04-Exporting-ranking-models.ipynb) in the Merlin Models GH repo.

# In[11]:


model = mm.DLRMModel(
    schema,
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(target_column)
)


# In[12]:


model.compile(optimizer='adam', run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
model.fit(train, validation_data=valid, batch_size=16*1024)


# We will create the feature repo in the current working directory, which is `BASE_DIR` for us.

# In[13]:


# set up the base dir to for feature store
BASE_DIR = os.environ.get("BASE_DIR", "/Merlin/examples/Building-and-deploying-multi-stage-RecSys/")


# Let's save our DLRM model to be able to load back at the deployment stage. 

# In[14]:


model.save(os.path.join(BASE_DIR, 'dlrm'))


# ### Training a Retrieval Model with Two-Tower Model

# Now we move to the offline retrieval stage. We are going to train a Two-Tower model for item retrieval. To learn more about the Two-tower model you can visit [05-Retrieval-Model.ipynb](https://github.com/NVIDIA-Merlin/models/blob/main/examples/05-Retrieval-Model.ipynb).

# In[15]:


output_path = os.path.join(DATA_FOLDER, 'processed/retrieval')


# We select only positive interaction rows where `click==1` in the dataset with `Filter()` operator.

# In[16]:


user_id = ["user_id"] >> Categorify(dtype='int32') >> TagAsUserID()
item_id = ["item_id"] >> Categorify(dtype='int32') >> TagAsItemID()

item_features = ["item_category", "item_shop", "item_brand"] >> Categorify(dtype='int32') >> TagAsItemFeatures()

user_features = ['user_shops', 'user_profile', 'user_group', 
       'user_gender', 'user_age', 'user_consumption_2', 'user_is_occupied',
       'user_geography', 'user_intentions', 'user_brands', 'user_categories'] \
        >> Categorify(dtype='int32') >> TagAsUserFeatures() 

inputs = user_id + item_id + item_features + user_features + ['click'] 

outputs = inputs >> Filter(f=lambda df: df["click"] == 1)

transform_aliccp((train, valid), output_path, nvt_workflow=outputs, workflow_name='workflow_retrieval')


# In[17]:


train_tt = Dataset(os.path.join(output_path, 'train', '*.parquet'))
valid_tt = Dataset(os.path.join(output_path, 'valid', '*.parquet'))

schema = train_tt.schema
schema = schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER])


# In[18]:


model = mm.TwoTowerModel(
    schema,
    query_tower=mm.MLPBlock([128, 64], no_activation_last_layer=True),        
    samplers=[mm.InBatchSampler()],
    embedding_options = mm.EmbeddingOptions(infer_embedding_sizes=True),
)


# In[19]:


model.compile(
    optimizer='adam', 
    run_eagerly=False, 
    loss="categorical_crossentropy", 
    metrics=[mm.RecallAt(10), mm.NDCGAt(10)]
)
model.fit(train_tt, validation_data=valid_tt, batch_size=1024*8, epochs=1)


# In the following cells we are going to export the required user and item features files, and save the query (user) tower model and item embeddings to disk. If you want to read more about exporting retrieval models, please visit [05-Retrieval-Model.ipynb](https://github.com/NVIDIA-Merlin/models/blob/main/examples/05-Retrieval-Model.ipynb) notebook in Merlin Models library repo.

# ### Set up a feature store with Feast

# Before we move onto the next step, we need to create a Feast feature repository.

# In[20]:


get_ipython().system('cd $BASE_DIR && feast init feature_repo')


# You should be seeing a message like <i>Creating a new Feast repository in ... </i> printed out above. Now, navigate to the `feature_repo` folder and remove the demo parquet file created by default, and `examples.py` file.

# In[21]:


os.remove(os.path.join(BASE_DIR, 'feature_repo', 'example.py'))
os.remove(os.path.join(BASE_DIR, 'feature_repo/data', 'driver_stats.parquet'))


# ### Exporting query (user) model

# In[22]:


query_tower = model.retrieval_block.query_block()
query_tower.save(os.path.join(BASE_DIR, 'query_tower'))


# ### Exporting user and item features

# In[23]:


from merlin.models.utils.dataset import unique_rows_by_features
user_features = unique_rows_by_features(train, Tags.USER, Tags.USER_ID).compute().reset_index(drop=True)


# In[24]:


user_features.head()


# We will artificially add `datetime` and `created` timestamp columns to our user_features dataframe. This required by Feast to track the user-item features and their creation time and to determine which version to use when we query Feast.

# In[25]:


from datetime import datetime
user_features["datetime"] = datetime.now()
user_features["datetime"] = user_features["datetime"].astype("datetime64[ns]")
user_features["created"] = datetime.now()
user_features["created"] = user_features["created"].astype("datetime64[ns]")


# In[26]:


user_features.head()


# In[27]:


user_features.to_parquet(os.path.join(BASE_DIR, 'feature_repo/data', 'user_features.parquet'))


# In[28]:


item_features = unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID).compute().reset_index(drop=True)


# In[29]:


item_features.shape


# In[30]:


item_features["datetime"] = datetime.now()
item_features["datetime"] = item_features["datetime"].astype("datetime64[ns]")
item_features["created"] = datetime.now()
item_features["created"] = item_features["created"].astype("datetime64[ns]")


# In[31]:


item_features.head()


# In[32]:


# save to disk
item_features.to_parquet(os.path.join(BASE_DIR, 'feature_repo/data', 'item_features.parquet'))


# ### Extract and save Item embeddings

# In[33]:


item_embs = model.item_embeddings(Dataset(item_features, schema=schema), batch_size=1024)
item_embs_df = item_embs.compute(scheduler="synchronous")


# In[34]:


# select only item_id together with embedding columns 
item_embeddings = item_embs_df.drop(columns=['item_category', 'item_shop', 'item_brand'])


# In[35]:


item_embeddings.head()


# In[36]:


# save to disk
item_embeddings.to_parquet(os.path.join(BASE_DIR,'item_embeddings.parquet'))


# ### Create feature definitions 

# Now we will create our user and item features definitions in the user_features.py and item_features.py files and save these files in the feature_repo.

# In[37]:


file = open(os.path.join(BASE_DIR, 'feature_repo/','user_features.py'), "w")
file.write(
'''
from google.protobuf.duration_pb2 import Duration
import datetime 
from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource

user_features = FileSource(
    path="{}",
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

user = Entity(name="user_id", value_type=ValueType.INT32, description="user id",)

user_features_view = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=Duration(seconds=86400 * 7),
    features=[
        Feature(name="user_shops", dtype=ValueType.INT32),
        Feature(name="user_profile", dtype=ValueType.INT32),
        Feature(name="user_group", dtype=ValueType.INT32),
        Feature(name="user_gender", dtype=ValueType.INT32),
        Feature(name="user_age", dtype=ValueType.INT32),
        Feature(name="user_consumption_2", dtype=ValueType.INT32),
        Feature(name="user_is_occupied", dtype=ValueType.INT32),
        Feature(name="user_geography", dtype=ValueType.INT32),
        Feature(name="user_intentions", dtype=ValueType.INT32),
        Feature(name="user_brands", dtype=ValueType.INT32),
        Feature(name="user_categories", dtype=ValueType.INT32),
    ],
    online=True,
    input=user_features,
    tags=dict(),
)
'''.format(os.path.join(BASE_DIR, 'feature_repo/data/','user_features.parquet'))
)
file.close()


# In[38]:


with open(os.path.join(BASE_DIR, 'feature_repo/','item_features.py'), "w") as f:
    f.write(
'''
from google.protobuf.duration_pb2 import Duration
import datetime 
from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource

item_features = FileSource(
    path="{}",
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

item = Entity(name="item_id", value_type=ValueType.INT32, description="item id",)

item_features_view = FeatureView(
    name="item_features",
    entities=["item_id"],
    ttl=Duration(seconds=86400 * 7),
    features=[
        Feature(name="item_category", dtype=ValueType.INT32),
        Feature(name="item_shop", dtype=ValueType.INT32),
        Feature(name="item_brand", dtype=ValueType.INT32),
    ],
    online=True,
    input=item_features,
    tags=dict(),
)
'''.format(os.path.join(BASE_DIR, 'feature_repo/data/','item_features.parquet'))
    )
file.close() 


# Let's checkout our Feast feature repository structure.

# In[ ]:


# install seedir
get_ipython().system('pip install seedir')


# In[41]:


import seedir as sd
feature_repo_path = os.path.join(BASE_DIR, 'feature_repo')
sd.seedir(feature_repo_path, style='lines', itemlimit=10, depthlimit=3, exclude_folders='.ipynb_checkpoints', sort=True)


# ### Next Steps
# We trained and exported our ranking and retrieval models and NVTabular workflows. In the next step, we will learn how to deploy our trained models into [Triton Inference Server (TIS)](https://github.com/triton-inference-server/server) with Merlin Systems library.
# 
# For the next step, move on to the `02-Deploying-multi-stage-Recsys-with-Merlin-Systems.ipynb` notebook to deploy our saved models as an ensemble to TIS and obtain prediction results for a given request.
