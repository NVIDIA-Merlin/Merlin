# Deploying a Ranking model on Triton Inference Server
The last step of ML pipeline is to deploy the trained model on [Triton Inference Server](https://github.com/triton-inference-server/server). NVIDIA Triton™, an open-source inference serving software, standardizes AI model deployment and execution and delivers fast and scalable AI in production. Merlin Systems library is designed for building pipelines to generate recommendations. Deploying pipelines on Triton is one part of the library's functionality and it provides easy to use API to be able to geneate 

The `ranking.py` is a template script that leverages the Merlin [models](https://github.com/NVIDIA-Merlin/models/) library (Tensorflow API) to build, train, evaluate ranking models. In the end you can either save the model for interence or persist model predictions to file.

Merlin models library provides building blocks on top of Tensorflow (Keras) that make it easy to build and train advanced neural ranking models. There are blocks for representing input, configuring model outputs/heads, perform feature interactions, losses, metrics, negative sampling, among others.



## Creating the Ensemble Graph

## Launching Triton Inference Server
It is common to find scenarios where you need to score the likelihood of different user-item events, e.g., clicking, liking, sharing, commenting, following the author, etc. Multi-Task Learning (MTL) techniques have been popular in deep learning to train a single model that is able to predict multiple targets at the same time.

By using MTL, it is typically possible to improve the tasks accuracy for somewhat correlated tasks, in particular for sparser targets, for which less training data is available. And instead of spending computational resources to train and deploy different models for each task, you can train and deploy a single MTL model that is able to predict multiple targets.

You can find more details in this [post](https://medium.com/nvidia-merlin/building-ranking-models-powered-by-multi-task-learning-with-merlin-and-tensorflow-4b4f993f7cc3) on the multi-task learning building blocks provided by [models](https://github.com/NVIDIA-Merlin/models/)  library.

The `ranking.py` script makes it easy to use multi-task learning backed by models library. It is automatically enabled when you provide more than one target column to `--tasks` argument.


<center>
<img src="https://miro.medium.com/v2/resize:fit:720/0*Fo6rIr10IJQCB6sb" alt="Multi-task learning architectures" >
</center>


## Command line arguments

In this section we describe the command line arguments of the `inference.py` script.

> You can check how to [setup the Docker image](../../ranking.md) to run `ranking.py` script with Docker.

This is an example command line for running the training for the TenRec dataset in our Docker image, which is explained [here](../../ranking.md).
 The parameters and their values can be separated by either space or by `=`.


```bash
cd /Merlin/examples/quick_start/scripts/inference/
OUT_DATASET_PATH=/outputs/dataset
CUDA_VISIBLE_DEVICES=0 TF_GPU_ALLOCATOR=cuda_malloc_async python  inference.py -- ....
```

### Inputs

```
  --train_data_path
                        Path of the train set. It expects a folder with parquet files.
                        If not provided, the model will not be trained (in case you want to use
                        --load_model_path to load a pre-trained model)
  --eval_data_path
                        Path of the eval set. It expects a folder with parquet files.
                        If not provided, the model will not be evaluated
  --predict_data_path
                        Path of a dataset for prediction. It expects a folder with parquet files
                        If provided, it will compute the predictions for this dataset and
                        save those predictions to --predict_output_path
  --load_model_path     
                        If provided, loads a model saved by --save_model_path
                        instead of initializing the parameters randomly
```


### Model
```
  --model {mmoe,cgc,ple,dcn,dlrm,mlp,wide_n_deep,deepfm}
                        Types of ranking model architectures that are
                        supported. Any of these models can be used with
                        multi-task learning (MTL). But these three are
                        specific to MTL: 'mmoe', 'cgc' and 'ple'. By default
                        'mlp'
  --activation 
                        Activation function supported by Keras, like:
                        'relu', 'selu', 'elu', 'tanh', 'sigmoid'. By
                        default 'relu'
  --mlp_init            Keras initializer for MLP layers. 
                        By default 'glorot_uniform'.
  --l2_reg              L2 regularization factor. By default 1e-5.
  --embeddings_l2_reg 
                        L2 regularization factor for embedding tables.
                        It operates only on the embeddings in the
                        current batch, not on the whole embedding table.
                        By default 0.0
  --embedding_sizes_multiplier 
                        When --embedding_dim is not provided it infers
                        automatically the embedding dimensions from the
                        categorical features cardinality. This factor
                        allows to increase/decrease the embedding dim
                        based on the cardinality. Typical values range
                        between 2 and 10. By default 2.0
  --dropout             Dropout rate. By default 0.0
  --mlp_layers 
                        Comma-separated dims of MLP layers. 
                        It is used by MLP model and also for dense blocks
                        of DLRM, DeepFM, DCN and Wide&Deep.
                        By default '128,64,32'
  --stl_positive_class_weight 
                        Positive class weight for single-task  models. By
                        default 1.0. The negative class weight is fixed
                        to 1.0
```


### Outputs
```
  --output_path
                        Folder to save training and logging assets.
  --save_model_path 
                        If provided, model is saved to this path after
                        training. It can be loaded later with --load_model_path 
  --predict_output_path 
                        If provided the prediction scores will be saved
                        to this path, according to --predict_output_format
                        and --predict_output_keep_cols 
  --predict_output_keep_cols 
                        Comma-separated list of columns to keep in the
                        output prediction file. If no columns is
                        provided, all columns are kept together with the
                        prediction scores.  
  --predict_output_format {parquet,csv,tsv}
                        Format of the output prediction files. By
                        default 'parquet', which is the most performant
                        format.
```                        