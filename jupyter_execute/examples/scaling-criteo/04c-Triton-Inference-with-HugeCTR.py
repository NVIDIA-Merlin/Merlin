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


# # Scaling Criteo: Triton Inference with HugeCTR
# 
# ## Overview
# 
# The last step is to deploy the ETL workflow and saved model to production. In the production setting, we want to transform the input data as during training (ETL). We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integer before we use the deep learning model for a prediction. Therefore, we deploy the NVTabular workflow with the HugeCTR model as an ensemble model to Triton Inference. The ensemble model garantuees that the same transformation are applied to the raw inputs.
# 
# <img src='./imgs/triton-hugectr.png' width="25%">
# 
# ### Learning objectives
# 
# In this notebook, we learn how to deploy our models to production
# - Use **NVTabular** to generate config and model files for Triton Inference Server
# - Deploy an ensemble of NVTabular workflow and HugeCTR model
# - Send example request to Triton Inference Server

# ## Getting Started
# 
# First, we need to generate the Triton Inference Server configurations and save the models in the correct format. In the previous notebooks [02-ETL-with-NVTabular](./02-ETL-with-NVTabular.ipynb) and [03c-Training-with-HugeCTR](./03c-Training-with-HugeCTR.ipynb) we saved the NVTabular workflow and HugeCTR model to disk. We will load them.

# ### Saving Ensemble Model for Triton Inference Server

# After training terminates, we can see that two `.model` files are generated. We need to move them inside a temporary folder, like `criteo_hugectr/1`. Let's create these folders.

# In[2]:


import os
import numpy as np


# In[3]:


os.system("mkdir -rf ./criteo_hugectr/")
os.system("mkdir -p ./criteo_hugectr/1")


# Now we move our saved `.model` files inside 1 folder. We use only the last snapshot after `9600` iterations.

# In[4]:


os.system("mv *9600.model ./criteo_hugectr/1/")


# Now we have to create a JSON file for inference which has a similar configuration as our training file. We should remove the solver and optimizer clauses and add the inference clause in the JSON file. The paths of the stored dense model and sparse model(s) should be specified at dense_model_file and sparse_model_file within the inference clause. We need to make some modifications to data in the layers clause. Besides, we need to change the last layer from BinaryCrossEntropyLoss to Sigmoid. The rest of "layers" should be exactly the same as that in the training model.py file.

# In[5]:


get_ipython().run_cell_magic('writefile', "'./criteo_hugectr/1/criteo.json'", '\n{\n   "inference": {\n    "max_batchsize": 64,\n    "hit_rate_threshold": 0.6,\n    "dense_model_file": "/models/criteo/1/_dense_9600.model",\n    "sparse_model_file": "/models/criteo/1/0_sparse_9600.model",\n    "label": 1,\n    "input_key_type": "I64"\n  },\n  "layers": [\n    {\n      "name": "data",\n      "type": "Data",\n      "format": "Parquet",\n      "slot_size_array": [10000000, 10000000, 3014529, 400781, 11, 2209, 11869, 148, 4, 977, 15, 38713, 10000000, 10000000, 10000000, 584616, 12883, 109, 37, 17177, 7425, 20266, 4, 7085, 1535, 64],\n      "source": "/raid/data/criteo/test_dask/output/train/_file_list.txt",\n      "eval_source": "/raid/data/criteo/test_dask/output/valid/_file_list.txt",\n      "check": "None",\n      "label": {\n        "top": "label",\n        "label_dim": 1\n      },\n      "dense": {\n        "top": "dense",\n        "dense_dim": 13\n      },\n      "sparse": [\n        {\n            "top": "data1",\n            "type": "LocalizedSlot",\n            "max_feature_num_per_sample": 26,\n            "max_nnz": 1,\n            "slot_num": 26\n        }\n      ]\n    },\n    {\n          "name": "sparse_embedding1",\n          "type": "LocalizedSlotSparseEmbeddingHash",\n          "bottom": "data1",\n          "top": "sparse_embedding1",\n          "sparse_embedding_hparam": {\n            "max_vocabulary_size_per_gpu": 15500000,\n            "embedding_vec_size": 128,\n            "combiner": 0\n          }\n      },\n      {\n        "name": "fc1",\n        "type": "InnerProduct",\n        "bottom": "dense",\n        "top": "fc1",\n         "fc_param": {\n          "num_output": 512\n        }\n      },\n  \n      {\n        "name": "relu1",\n        "type": "ReLU",\n        "bottom": "fc1",\n        "top": "relu1" \n      },\n  \n      {\n        "name": "fc2",\n        "type": "InnerProduct",\n        "bottom": "relu1",\n        "top": "fc2",\n         "fc_param": {\n          "num_output": 256\n        }\n      },\n  \n      {\n        "name": "relu2",\n        "type": "ReLU",\n        "bottom": "fc2",\n        "top": "relu2"     \n      },\n      \n      {\n        "name": "fc3",\n        "type": "InnerProduct",\n        "bottom": "relu2",\n        "top": "fc3",\n         "fc_param": {\n          "num_output": 128\n        }\n      },\n  \n      {\n        "name": "relu3",\n        "type": "ReLU",\n        "bottom": "fc3",\n        "top": "relu3"     \n      },\n    {\n        "name": "interaction1",\n        "type": "Interaction",\n        "bottom": ["relu3", "sparse_embedding1"],\n        "top": "interaction1"\n    },\n    {\n        "name": "fc4",\n        "type": "InnerProduct",\n        "bottom": "interaction1",\n        "top": "fc4",\n         "fc_param": {\n          "num_output": 1024\n        }\n      },\n      {\n        "name": "relu4",\n        "type": "ReLU",\n        "bottom": "fc4",\n        "top": "relu4" \n      },\n      {\n        "name": "fc5",\n        "type": "InnerProduct",\n        "bottom": "relu4",\n        "top": "fc5",\n         "fc_param": {\n          "num_output": 1024\n        }\n      },\n      {\n        "name": "relu5",\n        "type": "ReLU",\n        "bottom": "fc5",\n        "top": "relu5"     \n      },\n      {\n        "name": "fc6",\n        "type": "InnerProduct",\n        "bottom": "relu5",\n        "top": "fc6",\n         "fc_param": {\n          "num_output": 512\n        }\n      },\n      {\n        "name": "relu6",\n        "type": "ReLU",\n        "bottom": "fc6",\n        "top": "relu6"     \n      },\n      {\n        "name": "fc7",\n        "type": "InnerProduct",\n        "bottom": "relu6",\n        "top": "fc7",\n         "fc_param": {\n          "num_output": 256\n        }\n      },\n      {\n        "name": "relu7",\n        "type": "ReLU",\n        "bottom": "fc7",\n        "top": "relu7"     \n      },\n      {\n        "name": "fc8",\n        "type": "InnerProduct",\n        "bottom": "relu7",\n        "top": "fc8",\n         "fc_param": {\n          "num_output": 1\n        }\n      },\n    {\n      "name": "sigmoid",\n      "type": "Sigmoid",\n      "bottom": "fc8",\n      "top": "sigmoid"\n    } \n  ]\n}\n')


# Now we can save our models to be deployed at the inference stage. To do so we will use export_hugectr_ensemble method below. With this method, we can generate the config.pbtxt files automatically for each model. In doing so, we should also create a hugectr_params dictionary, and define the parameters like where the amazonreview.json file will be read, slots which corresponds to number of categorical features, `embedding_vector_size`, `max_nnz`, and `n_outputs` which is number of outputs.<br><br>
# The script below creates an ensemble triton server model where
# - workflow is the the nvtabular workflow used in preprocessing,
# - hugectr_model_path is the HugeCTR model that should be served. 
# - This path includes the .model files.name is the base name of the various triton models
# - output_path is the path where is model will be saved to.
# - cats are the categorical column names
# - conts are the continuous column names

# We need to load the NVTabular workflow first

# In[6]:


import nvtabular as nvt

BASE_DIR = os.environ.get("BASE_DIR", "/raid/data/criteo")
input_path = os.path.join(BASE_DIR, "test_dask/output")
workflow = nvt.Workflow.load(os.path.join(input_path, "workflow"))


# Let's clear the directory

# In[7]:


os.system("rm -rf /models/*")


# In[8]:


from nvtabular.inference.triton import export_hugectr_ensemble

hugectr_params = dict()
hugectr_params["config"] = "/models/criteo/1/criteo.json"
hugectr_params["slots"] = 26
hugectr_params["max_nnz"] = 1
hugectr_params["embedding_vector_size"] = 128
hugectr_params["n_outputs"] = 1
export_hugectr_ensemble(
    workflow=workflow,
    hugectr_model_path="./criteo_hugectr/1/",
    hugectr_params=hugectr_params,
    name="criteo",
    output_path="/models/",
    label_columns=["label"],
    cats=["C" + str(x) for x in range(1, 27)],
    conts=["I" + str(x) for x in range(1, 14)],
    max_batch_size=64,
)


# We can take a look at the generated files.

# In[9]:


get_ipython().system('tree /models')


# ### Loading Ensemble Model with Triton Inference Server
# 
# We have only saved the models for Triton Inference Server. We started Triton Inference Server in explicit mode, meaning that we need to send a request that Triton will load the ensemble model.

# We connect to the Triton Inference Server.

# In[11]:


import tritonhttpclient

try:
    triton_client = tritonhttpclient.InferenceServerClient(url="triton:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))


# We deactivate warnings.

# In[12]:


import warnings

warnings.filterwarnings("ignore")


# We check if the server is alive.

# In[13]:


triton_client.is_server_live()


# We check the available models in the repositories:
# - criteo_ens: Ensemble 
# - criteo_nvt: NVTabular 
# - criteo: HugeCTR model

# In[14]:


triton_client.get_model_repository_index()


# We load the models individually.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="criteo_nvt")\n')


# In[19]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="criteo")\n')


# In[17]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="criteo_ens")\n')


# ### Example Request to Triton Inference Server
# 
# Now, the models are loaded and we can create a sample request. We read an example **raw batch** for inference.

# In[20]:


import cudf

# read in the workflow (to get input/output schema to call triton with)
batch_path = os.path.join(BASE_DIR, "converted/criteo")
batch = cudf.read_parquet(os.path.join(batch_path, "*.parquet"), num_rows=3)
batch = batch[[x for x in batch.columns if x != "label"]]
print(batch)


# We prepare the batch for inference by using correct column names and data types. We use the same datatypes as defined in our dataframe.

# In[21]:


batch.dtypes


# In[22]:


import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

inputs = []

col_names = list(batch.columns)
col_dtypes = [np.int32] * len(col_names)

for i, col in enumerate(batch.columns):
    d = batch[col].values_host.astype(col_dtypes[i])
    d = d.reshape(len(d), 1)
    inputs.append(httpclient.InferInput(col_names[i], d.shape, np_to_triton_dtype(col_dtypes[i])))
    inputs[i].set_data_from_numpy(d)


# We send the request to the triton server and collect the last output.

# In[23]:


# placeholder variables for the output
outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

# build a client to connect to our server.
# This InferenceServerClient object is what we'll be using to talk to Triton.
# make the request with tritonclient.http.InferInput object
response = triton_client.infer("criteo_ens", inputs, request_id="1", outputs=outputs)

print("predicted sigmoid result:\n", response.as_numpy("OUTPUT0"))


# Let's unload the model. We need to unload each model.

# In[24]:


triton_client.unload_model(model_name="criteo_ens")
triton_client.unload_model(model_name="criteo_nvt")
triton_client.unload_model(model_name="criteo")

