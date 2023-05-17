# Deploying a Ranking model on Triton Inference Server
The last step of ML pipeline is to deploy the trained model into production. For this purpose we use [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server), which is an open-source inference serving software, standardizes AI model deployment and execution and delivers fast and scalable AI in production. 

[Merlin Systems](https://github.com/NVIDIA-Merlin/systems/tree/main) library is designed for building pipelines to generate recommendations. Deploying pipelines on Triton is one part of the library's functionality and Merlin Systems provides easy to use APIs to be able to export ensemble graph and model artifacts so that they can be loaded on Triton with less effort.

In this example we demonstrate the necessary steps to deploy a model to Triton and test it:

1. Creating the ensemble graph
2. Launching the Triton Inference Server
3. Sending request to Server and receiving the response

## Creating the Ensemble Graph

In order to do model deployment stage, you are required to complete `preprocessing` and `ranking` steps already from the [Quick-start for Ranking](../ranking.md).  At the inference step, we might have a collection of multiple (individual) models to be deployed on Triton. In this example, we deploy our NVTabular workflow model to be able to transform raw data the same way as in the dataset preprocessing phase, in order to avoid the training-serving skew. 

In this context, deploying multiple models is called an ensemble model since it represents a pipeline of one or more models that are sequentially connected, i.e., output of a model is the input of next model. Ensemble models are intended to be used to encapsulate a procedure that involves multiple models, such as "data preprocessing -> inference -> data postprocessing". 

The Triton Inference Server serves models from one or more model repositories that are specified when the server is started. Each model must include a configuration that provides required and optional information about the model. Merlin Systems simplified that step, so that we can easily export ensemble graph config files and artifacts. We use [Ensemble](https://github.com/NVIDIA-Merlin/systems/blob/main/merlin/systems/dag/ensemble.py#L29) class for that, which is responsible for interpreting the graph and exporting the correct files for the Triton server.

Exporting an ensemble graph consists of the following steps:

- loading saved workflow
- loading saved ranking model
- generating ensemble graph
- exporting the ensemble graph models and artifacts

These steps are taken care of by `inference.py` script when executed (please see the `Command line arguments` section below for the instructions).


## Launching Triton Inference Server

Once the models ensemble graph is exported to the path that you define, now you can load these models on Triton Inference Server, which is actually only one single line of code. 

You can start the server by running the following command:

```bash
tritonserver --model-repository=<path to the saved ensemble folder>
```
For the `--model-repository` argument, provide the same path of `ensemble_export_path` argument that you inputted previously when executing the `inference.py` script.

After you run the tritonserver command, wait until your terminal shows messages like the following example:

I0414 18:29:50.741833 4067 grpc_server.cc:4421] Started GRPCInferenceService at 0.0.0.0:8001
I0414 18:29:50.742197 4067 http_server.cc:3113] Started HTTPService at 0.0.0.0:8000
I0414 18:29:50.783470 4067 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002 ,br>


## Sending request to Triton

This step is explained and demonstrated in the [inference.ipynb](https://github.com/NVIDIA-Merlin/Merlin/blob/quick_start_inf_triton/examples/quick_start/scripts/inference/inference.ipynb) example notebook. Please follow the instructions there and execute the cells to send a request and receive response from Triton.

## Command line arguments
In this section we describe the command line arguments of the `inference.py` script.

This is an example command line for running the `inference.py`script after your finished model `preprocessing` and `ranking` steps.

```bash
cd /Merlin/examples/quick_start/scripts/inference/
NVT_WORKFLOW_PATH=<input path with saved workflow>
TF_SAVED_MODEL_PATH=<input path with saved model>
OUTPUT_ENSEMBLE_PATH=<output path to export the Triton ensemble model>
TF_GPU_ALLOCATOR=cuda_malloc_async python inference.py --nvt_workflow_path $NVT_WORKFLOW_PATH --load_model_path $TF_SAVED_MODEL_PATH --ensemble_export_path $OUTPUT_ENSEMBLE_PATH
```

Note that preprocessing step saves the NVTabular workflow automatically to `output_path` that is set when executing preprocessing script. For the `load_model_path` argument, be sure that you provide the exact same path f that you provided for saving the trained model during ranking step.

### Inputs

```
  --nvt_workflow_path   
                        Loads the nvtabular workflow saved in the preprocessing step (`--output_path`).
  --load_model_path     
                        Loads a model saved by --save_model_path in the ranking step.
   --ensemble_export_path
                        Path for exporting the config files and model artifacts
                        to load them on Triton inference server.
```
             