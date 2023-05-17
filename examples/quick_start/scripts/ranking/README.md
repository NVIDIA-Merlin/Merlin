# Ranking script
The `ranking.py` is a template script that leverages the Merlin [models](https://github.com/NVIDIA-Merlin/models/) library (Tensorflow API) to build, train, evaluate ranking models. In the end you can either save the model for interence or persist model predictions to file.

Merlin models library provides building blocks on top of Tensorflow (Keras) that make it easy to build and train advanced neural ranking models. There are blocks for representing input, configuring model outputs/heads, perform feature interactions, losses, metrics, negative sampling, among others.

## Ranking in multi-stage RecSys
Large online services like social media, streaming, e-commerce, and news provide a very broad catalog of items and leverage recommender systems to help users find relevant items. Those companies typically deploy recommender systems pipelines with [multiple stages](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e), in particular the retrieval and ranking. The retrieval stage selects a few hundreds or thousands of items from a large catalog. It can be a heuristic approach (like most recent items) or a scalable model like Matrix Factorization, [Two-Tower architecture](https://medium.com/nvidia-merlin/scale-faster-with-less-code-using-two-tower-with-merlin-c16f32aafa9f) or [YouTubeDNN](https://static.googleusercontent.com/media/research.google.com/pt-BR//pubs/archive/45530.pdf). Then, the ranking stage scores the relevance of the candidate items provided by the previous stage for a given user and context.


## Multi-task learning for ranking models
It is common to find scenarios where you need to score the likelihood of different user-item events, e.g., clicking, liking, sharing, commenting, following the author, etc. Multi-Task Learning (MTL) techniques have been popular in deep learning to train a single model that is able to predict multiple targets at the same time.

By using MTL, it is typically possible to improve the tasks accuracy for somewhat correlated tasks, in particular for sparser targets, for which less training data is available. And instead of spending computational resources to train and deploy different models for each task, you can train and deploy a single MTL model that is able to predict multiple targets.

You can find more details in this [post](https://medium.com/nvidia-merlin/building-ranking-models-powered-by-multi-task-learning-with-merlin-and-tensorflow-4b4f993f7cc3) on the multi-task learning building blocks provided by [models](https://github.com/NVIDIA-Merlin/models/)  library.

The `ranking.py` script makes it easy to use multi-task learning backed by models library. It is automatically enabled when you provide more than one target column to `--tasks` argument.

## Supported models
The `ranking.py` script makes it easy to use baseline and advanced deep ranking models available in [models](https://github.com/NVIDIA-Merlin/models/) library.  
The script can be also used as an **advanced example** that demonstrate [how to set specific hyperparameters using models API](ranking_models.py).

### Baseline ranking architectures
- **MLP** (`--model=mlp`)  - Simple multi-layer perceptron architecture. More info in [MLPBlock](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.MLPBlock.html#merlin.models.tf.MLPBlock). 
- **Wide and Deep** - Aims to leverage the ability of neural networks to generalize and capacity of linear models to memorize relevant feature interactions. The deep part is an MLP model, with categorical features represented as embeddings, which are concatenated with continuous features and fed through multiple MLP layers. The wide part is a linear model takes a sparse representation of categorical features (i.e. one-hot or multi-hot representation). More info in [WideAndDeepModel](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.WideAndDeepModel.html#merlin.models.tf.WideAndDeepModel) and its  [paper](https://dl.acm.org/doi/10.1145/2988450.2988454).


- **DeepFM** (`--model=deepfm`) - DeepFM architecture is a combination of a Factorization Machine and a Deep Neural Network. More info in [DeepFMModel](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.DeepFMModel.html#merlin.models.tf.DeepFMModel) and its [paper](https://arxiv.org/abs/1703.04247).
- **DRLM** (`--model=dlrm`) - Continuous features are concatenated and combined by the bottom MLP to produce an embedding like categorical embeddings. The factorization machines layer perform 2nd level feature interaction of those embeddings, which need to have the same dim.  Then those outputs are concatenated and processed through the top MLP layer to output the predictions. More info in [DLRMModel](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.DLRMModel.html#merlin.models.tf.DLRMModel) and its [paper](https://arxiv.org/abs/1906.00091). 
- **DCN-v2** (`--model=dcn`) - The Improved Deep & Cross Network combines a MLP network with cross-network for powerful and bounded feature interaction. More info in [DCNModel](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.DCNModel.html#merlin.models.tf.DCNModel) and its [paper](https://dl.acm.org/doi/10.1145/3442381.3450078).

### Multi-task learning architectures
- **MMOE** (`--model=mmoe`) - The Multi-gate Mixture-of-Experts (MMoE) is one of the most popular models for multi-task learning on tabular data. It allows parameters to be automatically allocated to capture either shared task information or task-specific information. The core components of MMoE are experts and gates. Instead of using a shared-bottom for all tasks, it has multiple expert sub-networks processing input features independently from each other. Each task has an independent gate, which dynamically selects based on the inputs the level with which the task wants to leverage the output of each expert. More info on [MMOEBlock](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.MMOEBlock.html#merlin.models.tf.MMOEBlock) and its [paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007).
- **CGC** (`--model=cgc`) - Instead of having tasks sharing all the experts like in MMOE, it allows for splitting task-specific experts and shared experts, in an architecture named Customized Gate Control (CGC) Model. More info on [CGCBlock](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.CGCBlock.html#merlin.models.tf.CGCBlock) and its [paper](https://dl.acm.org/doi/10.1145/3383313.3412236).
- **PLE** (`--model=ple`) - In the same paper introducing CGC, authors proposed stacking multiple CGC models on top of each other to form a multi-level MTL model, so that the model can progressively combine shared and task-specific experts. They name this approach as Progressive Layered Extraction (PLE). Their paper experiments showed accuracy improvements by using PLE compared to CGC. More info on [PLEBlock](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.CGCBlock.html#merlin.models.tf.CGCBlock) and its [paper](https://dl.acm.org/doi/10.1145/3383313.3412236).

<center>
<img src="https://miro.medium.com/v2/resize:fit:720/0*Fo6rIr10IJQCB6sb" alt="Multi-task learning architectures" >
</center>


## Best practices

### Modeling inputs features
Neural networks operate on top of dense / continuous float inputs. Continuous features fit nicely into that format, but categorical features needs to be represented accordingly.
It is assumed that in the preprocessing the categorical features were encoded as contiguous ids. Then, they are typically be represented by the model using:
- **One-hot encoding** Sparse representation where each categorical value is represented by a binary feature with 1 only for the actual value. If the categorical feature contains a list of values, it can be encoded with multi-hot encoding, with 1s for all values in the list. This encoding is useful to represent low-cardinality categorical features or to provide input to linear models.
- **Embedding** - This encoding is very popular for deep neural networks. Each categorical value is mapped to a 1D continuous vector, that can be trainable or pre-trained. The embeddings are stored in embedding layers or tables, whose first dim in the cardinality of the categorical feature and 2nd dim is the embedding size. 

#### **Dealing with high-cardinality categorical features**  
We explain in the [Quick-start preprocessing documentation](../preproc/README.md) that large services might have categorical features with very high cardinality (e.g. order of hundreds of millions or higher), like user id or item id. They typically require a high memory to be stored (e.g. with embedding tables) or processed (e.g. with one-hot encoding). In addition, most of the categorical values are very infrequent, for which it is not possible to learn good embeddings. 

The [preprocessing documentation](../preproc/README.md) describes some options to deal with the high-cardinality features: **Frequency capping**, **Filtering out rows with infrequent values** and  **Hashing**.

You might also decide to keep the original high-cardinality of the categorical features for better personalization level and accuracy. 

> The embedding tables are typically responsible for most of the parameters of Recommender System models. For large scale systems, where the number of users and items is in the order of hundreds of millions, it is typically needed to use a distributed embeddings solution, so that embedding embedding tables can be sharded in multiple compute devices (e.g. GPU, CPU). 

<!--**TODO**: Add references to NVIDIA distributed embeddings solutions -->

#### **Defining the embedding size**

It is common sense that higher the cardinality of categorical feature the higher should be the embedding dimension, as its vector space gets more complex.

Models library uses by default a heuristic method that sets embedding sizes based on cardinality (implementation [here](https://github.com/NVIDIA-Merlin/models/blob/a5e392cbc575fe984c96ddcbce696e4b71b7073d/merlin/models/utils/schema_utils.py#L169)), which can be scaled by `--embedding_sizes_multiplier`. Models library API also allows setting [specific embedding dims](https://nvidia-merlin.github.io/models/main/generated/merlin.models.tf.Embeddings.html#merlin.models.tf.Embeddings) for each / all categorical features.

> Some models supported by this script (DLRM and DeepFM) require the embedding sizes of categorical features to be the same (`--embedding_dim`) because of their feature interaction approach based on Factorization Machines.

### Regularization
Neural networks typically require regularization in order to avoid overfitting, in particular if trained on small data or for many epochs that can make it memorize train set. This script provide typical regularization techniques like Dropout (`--dropout`) and L2 regularization of model parameters (`--l2_reg`) and embeddings (`--embeddings_l2_reg`).               

### Classes weights
Typically positive user interactions are just a small fraction of the items that were exposed to the users. That leads to class unbalanced targets.  
A common technique to deal with this problem in machine learning is to assign a higher weight to the loss for examples with infrequent targets - positive classes in this case.  
You might set the positive class weight for single-task learning models with `--stl_positive_class_weight` and for multi-task learning you can set the class weight for each target separately by using `--mtl_pos_class_weight_*`, where `*` must be replaced by the target name. In this case, the negative class weight is always 1.0.

### Negative sampling
If you have only positive interactions in your training data, you can use negative sampling to include synthetic negative examples in the training batch. The negative samples are generated by adding for each positive example N negative examples, keeping user features and replacing features of the target item by other item interacted by another users in the batch. You can easily set the number of negative examples for train (`--in_batch_negatives_train`) and evaluate (`--in_batch_negatives_eval`).  
This functionality require that user and item features are tagged accordingly, as explained in the [Quick-start preprocessing documentation](../preproc/README.md). 

### Multi-task learning

#### **Losses weights**  

You can balance the learning of the multiple tasks by setting losses weights when using multi-task learning. You can set them by providing `--mtl_loss_weight_*` for each task, where `*` must be replaced by the target name.

#### **Setting tasks sample space** 

Some targets might depend on other target columns for some datasets. For example, the preprocessed TenRec dataset have positive (=1) `like`, `follow`, and `share` events only if the user has also clicked on the item (`click=1`). 

You might want to model dependent tasks explicitly by setting the sample space, i.e., computing the loss of the dependent tasks only for examples where the dominant target is 1. That would make the dependent targets less sparser, as their value is always 0 when dominant target is 0.

The scripts allows setting the tasks sample space by using `--tasks_sample_space`, which accepts comma-separated values. The order of sample spaces should match the order of the `--tasks`. Empty value means the task will be trained in the entire space, i.e., loss computed for all examples in the dataset.   
For TenRec dataset, you could use `--tasks click,like,share,follow` and `--tasks_sample_space=,click,click,click`, meaning that the click task will be trained using the entire space and the other tasks will be trained only in click space.

> We have observed empirically that if you want a model to predict all tasks at the same time (e.g. the likelihood of a user to click-like-share a post), it is better to train all tasks using entire space. On the other hand, if you want to train a MTL model that predicts rarer events (e.g. add-to-cart, purchase) given prior events (e.g. click), then you typically get better accuracy on the dependent tasks training them in the dominant task space, while training the dominant task on entire space.

## Command line arguments

In this section we describe the command line arguments of the `ranking.py` script.

> You can check how to [setup the Docker image](../../ranking.md) to run `ranking.py` script with Docker.

This is an example command line for running the training for the TenRec dataset in our Docker image, which is explained [here](../../ranking.md).
 The parameters and their values can be separated by either space or by `=`.


```bash
cd /Merlin/examples/quick_start/scripts/ranking/
OUT_DATASET_PATH=/outputs/dataset
CUDA_VISIBLE_DEVICES=0 TF_GPU_ALLOCATOR=cuda_malloc_async python  ranking.py --train_data_path $OUT_DATASET_PATH/train --eval_data_path $OUT_DATASET_PATH/eval --output_path ./outputs/ --tasks=click --stl_positive_class_weight 3 --model dlrm --embeddings_dim 64 --l2_reg 1e-4 --embeddings_l2_reg 1e-6 --dropout 0.05 --mlp_layers 64,32  --lr 1e-4 --lr_decay_rate 0.99 --lr_decay_steps 100 --train_batch_size 65536 --eval_batch_size 65536 --epochs 1 --save_model_path ./saved_model
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

### Tasks
```
  --tasks               Columns (comma-sep) with the target columns to
                        be predicted. A regression/binary classification
                        head is created for each of the target columns.
                        If more than one column is provided, then multi-
                        task learning is used to combine the tasks
                        losses. If 'all' is provided, all columns tagged
                        as target in the schema are used as tasks. By
                        default 'all'
  --tasks_sample_space 
                        Columns (comma-sep) to be used as sample space
                        for each task. This list of columns should match
                        the order of columns in --tasks. Typically this
                        is used to explicitly model that the task event
                        (e.g. purchase) can only occur when another
                        binary event has already happened (e.g. click).
                        Then by setting for example
                        --tasks=click,purchase
                        --tasks_sample_space,click, you configure the
                        training to compute the purchase loss only for
                        examples with click=1, making the purchase
                        target less sparser.
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

### DCN-v2
```
  --dcn_interacted_layer_num
                        Number of interaction layers for DCN-v2
                        architecture. By default 1.
```

### DLRM and DeepFM
```
  --embeddings_dim 
                        Sets the embedding dim for all embedding columns
                        to be the same. This is only used for --model
                        'dlrm' and 'deepfm'
```

### Wide&Deep
```
  --wnd_hashed_cross_num_bins 
                        Used with Wide&Deep model. Sets the number of
                        bins for hashing feature interactions. By
                        default 10000.
  --wnd_wide_l2_reg 
                        Used with Wide&Deep model. Sets the L2 reg of
                        the wide/linear sub-network. By default 1e-5.
  --wnd_ignore_combinations 
                        Feature interactions to ignore. Separate feature
                        combinations with ',' and columns with ':'. For
                        example: --wnd_ignore_combinations='item_id:item
                        _category,user_id:user_gender'
```

### Wide&Deep and DeepFM
```            
  --multihot_max_seq_length
                        DeepFM and Wide&Deep support multi-hot
                        categorical features for the wide/linear sub-
                        network. But they require setting the maximum
                        list length, i.e., number of different multi-hot
                        values that can exist in a given example. By
                        default 5.
```

### MMOE
```
  --mmoe_num_mlp_experts 
                        Number of experts for MMOE. All of them are
                        shared by all the tasks. By default 4.
```

### CGC and PLE
```                        
  --cgc_num_task_experts 
                        Number of task-specific experts for CGC and PLE.
                        By default 1.
  --cgc_num_shared_experts 
                        Number of shared experts for CGC and PLE. By
                        default 2.
  --ple_num_layers 
                        Number of CGC modules to stack for PLE
                        architecture. By default 1.
```        

### Expert-based MTL models
```
  --expert_mlp_layers 
                        For MTL models (MMOE, CGC, PLE) sets the MLP
                        layers of experts. 
                        It expects a comma-separated list of layer dims.
                        By default '64'
  --gate_dim            Dimension of the gate dim MLP layer. By default
                        64
  --mtl_gates_softmax_temperature 
                        Sets the softmax temperature for the gates
                        output layer, that provides weights for the
                        weighted average of experts outputs. By default
                        1.0
```

### Multi-task learning models
```
  --use_task_towers 
                        Creates task-specific MLP tower before its head.
                        By default True.
  --tower_layers 
                        MLP architecture of task-specific towers. 
                        It expects a comma-separated list of layer dims.
                        By default '64'
```

### Negative sampling
```
  --in_batch_negatives_train 
                        If greater than 0, enables in-batch sampling,
                        providing this number of negative samples per
                        positive. This requires that your data contains
                        only positive examples, and that item features
                        are tagged accordingly in the schema, for
                        example, by setting --item_features in the
                        preprocessing script.
  --in_batch_negatives_eval 
                        Same as --in_batch_negatives_train for
                        evaluation.
```

### Training and evaluation
```
  --lr LR               Learning rate
  --lr_decay_rate 
                        Learning rate decay factor. By default 0.99
  --lr_decay_steps 
                        Learning rate decay steps. It decreases the LR
                        at this frequency, by default each 100 steps
  --train_batch_size 
                        Train batch size. By default 1024. Larger batch
                        sizes are recommended for better performance.
  --eval_batch_size 
                        Eval batch size. By default 1024. Larger batch
                        sizes are recommended for better performance.
  --epochs EPOCHS       Number of epochs. By default 1.
  --optimizer {adagrad,adam}
                        Optimizer. By default 'adam'
  --train_metrics_steps 
                        How often should train metrics be computed
                        during training. You might increase this number
                        to reduce the frequency and increase a bit the
                        training throughput. By default 10.
  --validation_steps 
                        If not predicting, logs the validation metrics
                        for this number of steps at the end of each
                        training epoch. By default 10.
  --random_seed 
                        Random seed for some reproducibility. By default
                        42.
  --train_steps_per_epoch 
                        Number of train steps per epoch. Set this for
                        quick debugging.
```

### Logging
```
  --metrics_log_frequency 
                        --How often metrics should be logged to
                        Tensorboard or Weights&Biases. By default each
                        50 steps.
  --log_to_tensorboard 
                        Enables logging to Tensorboard.
  --log_to_wandb 
                        Enables logging to Weights&Biases. This requires 
                        sign-up for a free Weights&Biases account at https://wandb.ai/
                        and providing an API key in the console you can get at 
                        https://wandb.ai/authorize
  --wandb_project 
                        Name of the Weights&Biases project to log
  --wandb_entity 
                        Name of the Weights&Biases team/org to log
  --wandb_exp_group 
                        Not used by the script. Just used to allow for
                        logging some info to organize experiments in
                        Weights&Biases
```


This requires sign-up for a free Weights&Biases account at https://wandb.ai/home "
        "and providing an API key in the console.

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