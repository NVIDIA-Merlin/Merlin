import argparse
from enum import Enum


class Task(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


class MtlArgsPrefix(Enum):
    POS_CLASS_WEIGHT_ARG_PREFIX = "mtl_pos_class_weight_"
    LOSS_WEIGHT_ARG_PREFIX = "mtl_loss_weight_"


INT_LIST_ARGS = ["mlp_layers", "expert_mlp_layers", "tower_layers"]
STR_LIST_ARGS = [
    "tasks",
    "tasks_sample_space",
    "predict_output_keep_cols",
    "wnd_ignore_combinations",
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse_dynamic_args(dyn_args):
    dyn_args_dict = dict([arg.replace("--", "").split("=") for arg in dyn_args])
    return dyn_args_dict


def parse_list_arg(value, vtype=str):
    # Used to allow providing empty string ("") as command line argument
    if value is None or value == "None":
        value = ""

    alist = list([vtype(v.strip()) for v in value.split(",") if v != ""])
    return alist


def parse_arguments():
    parser = build_arg_parser()
    args, dynamic_args = parser.parse_known_args()
    dynamic_args = parse_dynamic_args(dynamic_args)

    unknown_args = list(
        [
            arg
            for arg in dynamic_args
            if not any([arg.startswith(prefix.value) for prefix in MtlArgsPrefix])
        ]
    )
    if len(unknown_args) > 0:
        raise ValueError(f"Unrecognized arguments: {unknown_args}")

    new_args = AttrDict({**args.__dict__, **dynamic_args})

    # Parsing str args that contains lists of ints
    for a in INT_LIST_ARGS:
        new_args[a] = parse_list_arg(new_args[a], vtype=int)

    for a in STR_LIST_ARGS:
        new_args[a] = parse_list_arg(new_args[a])

    # logging.info(f"ARGUMENTS: {new_args}")

    return new_args


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Script for building, training and evaluating ranking models."
    )

    # Inputs
    parser.add_argument(
        "--train_data_path",
        default=None,
        help="Path of the train set. It expects a folder with parquet files. "
        "If not provided, the model will not be trained (in case you want to use "
        "--load_model_path to load a pre-trained model)",
    )
    parser.add_argument(
        "--eval_data_path",
        default=None,
        help="Path of the eval set. It expects a folder with parquet files. "
        "If not provided, the model will not be evaluated",
    )

    parser.add_argument(
        "--predict_data_path",
        default=None,
        help="Path of a dataset for prediction. It expects a folder with parquet files. "
        "If provided, it will compute the predictions for this dataset and "
        "save those predictions to --predict_output_path",
    )

    # Outputs
    parser.add_argument(
        "--output_path",
        default="./output/",
        help="Folder to save training assets and logging.",
    )

    parser.add_argument(
        "--save_model_path",
        default=None,
        help="If provided, model is saved to this path after training. "
        "It can be loaded later with --load_model_path ",
    )

    parser.add_argument(
        "--load_model_path",
        default=None,
        help="If provided, loads a model saved by --save_model_path "
        "instead of initializing the parameters randomly",
    )

    parser.add_argument(
        "--predict_output_keep_cols",
        default=None,
        help="Comma-separated list of columns to keep in the output "
        "prediction file. If no columns is provided, all columns "
        "are kept together with the prediction scores.",
    )
    parser.add_argument(
        "--predict_output_path",
        default=None,
        help="If provided the prediction scores will be saved to this path.",
    )

    parser.add_argument(
        "--predict_output_format",
        default="parquet",
        choices=["parquet", "csv", "tsv"],
        help="Format of the output prediction files. By default 'parquet', "
        "which is the most performant format.",
    )

    # Tasks
    parser.add_argument(
        "--tasks",
        default="all",
        help="Columns (comma-sep) with the target columns to be predicted. "
        "A regression/binary classification head is created for each of the target columns. "
        "If more than one column is provided, then multi-task learning is used to combine "
        "the tasks losses. If 'all' is provided, all columns tagged as target in the schema "
        "are used as tasks. By default 'all'",
    )
    parser.add_argument(
        "--tasks_sample_space",
        default="",
        help="Columns (comma-sep) to be used as sample space for each task. "
        "This list of columns should match the order of columns in --tasks. "
        "Typically this is used to explicitly model "
        "that the task event (e.g. purchase) can only occur when another binary event "
        " has already happened (e.g. click). "
        "Then by setting for example "
        "--tasks=click,purchase --tasks_sample_space,click, you configure the training "
        "to compute the purchase loss only for examples with click=1, making the "
        "purchase target less sparser.",
    )

    # Model
    parser.add_argument(
        "--model",
        default="mlp",
        choices=["mmoe", "cgc", "ple", "dcn", "dlrm", "mlp", "wide_n_deep", "deepfm"],
        help="Types of ranking model architectures that are supported. Any of these "
        "models can be used with multi-task learning (MTL). But these three are "
        "specific to MTL: mmoe, cgc and ple. By default 'mlp'",
    )

    parser.add_argument(
        "--activation",
        default="relu",
        help="Activation function supported by Keras, like: 'relu', 'selu', 'elu', "
        "'tanh', 'sigmoid'. By default 'relu'",
    )

    parser.add_argument(
        "--mlp_init",
        type=str,
        default="glorot_uniform",
        help="Keras initializer for MLP layers." "By default 'glorot_uniform'.",
    )
    parser.add_argument(
        "--l2_reg",
        default=1e-5,
        type=float,
        help="L2 regularization factor. By default 1e-5.",
    )
    parser.add_argument(
        "--embeddings_l2_reg",
        default=0.0,
        type=float,
        help="L2 regularization factor for embedding tables. "
        "It operates only on the embeddings in the current batch, not on the "
        "whole embedding table. By default 0.0",
    )

    # Embeddings args
    parser.add_argument(
        "--embedding_sizes_multiplier",
        default=2.0,
        type=float,
        help="When --embedding_dim is not provided "
        "it infers automatically the embedding dimensions from the categorical "
        "features cardinality. This factor allows to increase/decrease the "
        "embedding dim based on the cardinality. "
        "Typical values range between 2 and 10. By default 2.0",
    )

    # MLPs args
    parser.add_argument(
        "--dropout", default=0.0, type=float, help="Dropout rate. By default 0.0"
    )

    # hyperparams for STL models
    parser.add_argument(
        "--mlp_layers",
        default="128,64,32",
        type=str,
        help="The dims of MLP layers. It is used by MLP model and also for dense blocks "
        "of DLRM, DeepFM, DCN and Wide&Deep. By default '128,64,32'",
    )
    parser.add_argument(
        "--stl_positive_class_weight",
        default=1.0,
        type=float,
        help="Positive class weight for single-task models. By default 1.0. "
        "The negative class weight is fixed to 1.0",
    )

    # DCN
    parser.add_argument(
        "--dcn_interacted_layer_num",
        default=1,
        type=int,
        help="Number of interaction layers for DCN-v2 architecture. By default 1.",
    )

    # DLRM & DeepFM
    parser.add_argument(
        "--embeddings_dim",
        default=128,
        type=int,
        help="Sets the embedding dim for all embedding columns to be the same. "
        "This is only used for --model 'dlrm' and 'deepfm'",
    )

    # Wide&Deep
    parser.add_argument(
        "--wnd_hashed_cross_num_bins",
        default=10000,
        type=int,
        help="Used with Wide&Deep model. Sets the number of bins for hashing "
        "feature interactions. By default 10000.",
    )
    parser.add_argument(
        "--wnd_wide_l2_reg",
        default=1e-5,
        type=float,
        help="Used with Wide&Deep model. Sets the L2 reg of the wide/linear sub-network. "
        "By default 1e-5.",
    )
    parser.add_argument(
        "--wnd_ignore_combinations",
        default=None,
        type=str,
        help="Feature interactions to ignore. Separate feature combinations "
        "with ',' and columns with ':'. For example: "
        "--wnd_ignore_combinations='item_id:item_category,user_id:user_gender'",
    )

    # DeepFM & Wide&Deep
    parser.add_argument(
        "--multihot_max_seq_length",
        default=5,
        type=float,
        help="DeepFM and Wide&Deep support multi-hot categorical features for the wide/linear "
        "sub-network. But they require setting the maximum list length, i.e., number of different "
        "multi-hot values that can exist in a given example. By default 5.",
    )

    # MMOE
    parser.add_argument(
        "--mmoe_num_mlp_experts",
        default=4,
        type=int,
        help="Number of experts for MMOE. All of them are shared by all the tasks. "
        "By default 4.",
    )

    # CGC and PLE
    parser.add_argument(
        "--cgc_num_task_experts",
        default=1,
        type=int,
        help="Number of task-specific experts for CGC and PLE. By default 1.",
    )
    parser.add_argument(
        "--cgc_num_shared_experts",
        default=2,
        type=int,
        help="Number of shared experts for CGC and PLE. By default 2.",
    )
    parser.add_argument(
        "--ple_num_layers",
        default=1,
        type=int,
        help="Number of CGC modules to stack for PLE architecture. By default 1.",
    )

    # hyperparams for expert-based multi-task models (MMOE, CGC, PLE)

    parser.add_argument(
        "--expert_mlp_layers",
        default="64",
        type=str,
        help="For MTL models (MMOE, CGC, PLE) sets the MLP "
        "layers of experts.  It expects a comma-separated "
        "list of layer dims. By default '64'",
    )

    parser.add_argument(
        "--gate_dim",
        default=64,
        type=int,
        help="Dimension of the gate dim MLP layer. By default 64",
    )

    parser.add_argument(
        "--mtl_gates_softmax_temperature",
        default=1.0,
        type=float,
        help="Sets the softmax temperature for the gates output layer, "
        "that provides weights for the weighted average of experts outputs. "
        "By default 1.0",
    )

    parser.add_argument(
        "--use_task_towers",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enables task-specific towers before its head. By default True.",
    )

    parser.add_argument(
        "--tower_layers",
        default="64",
        type=str,
        help="MLP architecture of task-specific towers. "
        "It expects a comma-separated list of layer dims. By default '64'",
    )

    # hyperparams for training
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument(
        "--lr_decay_rate",
        default=0.99,
        type=float,
        help=" Learning rate decay factor. By default 0.99",
    )
    parser.add_argument(
        "--lr_decay_steps",
        default=100,
        type=int,
        help="Learning rate decay steps. It decreases the LR at this frequency, "
        "by default each 100 steps",
    )
    parser.add_argument(
        "--train_batch_size",
        default=1024,
        type=int,
        help=" Train batch size. By default 1024. Larger batch sizes are recommended "
        "for better performance.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=1024,
        type=int,
        help="Eval batch size. By default 1024. Larger batch sizes are recommended "
        "for better performance.",
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of epochs. By default 1."
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adagrad", "adam"],
        help="Optimizer. By default 'adam'",
    )

    parser.add_argument(
        "--train_metrics_steps",
        default=10,
        type=int,
        help="How often should train metrics be computed during training. "
        "You might increase this number to reduce the frequency and increase a bit the "
        "training throughput. By default 10.",
    )

    parser.add_argument(
        "--validation_steps",
        default=10,
        type=int,
        help="If not predicting, logs the validation metrics for "
        "this number of steps at the end of each training epoch. By default 10.",
    )

    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed for some reproducibility. By default 42.",
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        type=int,
        help="Number of train steps per epoch. Set this for quick debugging.",
    )

    # In-batch negatives
    parser.add_argument(
        "--in_batch_negatives_train",
        default=0,
        type=int,
        help="If greater than 0, enables in-batch sampling, providing this number of "
        "negative samples per positive. This requires that your data contains "
        "only positive examples, and that item features are tagged accordingly in the schema, "
        "for example, by setting --item_features in the preprocessing script.",
    )
    parser.add_argument(
        "--in_batch_negatives_eval",
        default=0,
        type=int,
        help="Same as --in_batch_negatives_train for evaluation.",
    )

    # Logging
    parser.add_argument(
        "--metrics_log_frequency",
        default=50,
        type=int,
        help="--How often metrics should be logged to Tensorboard or Weights&Biases. "
        "By default each 50 steps.",
    )
    parser.add_argument(
        "--log_to_tensorboard",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enables logging to Tensorboard.",
    )

    parser.add_argument(
        "--log_to_wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enables logging to Weights&Biases. "
        "This requires sign-up for a free Weights&Biases account at https://wandb.ai/home "
        "and providing an API key in the console you can get at https://wandb.ai/authorize",
    )

    parser.add_argument(
        "--wandb_project",
        default="mm_quick_start",
        help="Name of the Weights&Biases project to log",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="Name of the Weights&Biases team/org to log",
    )
    parser.add_argument(
        "--wandb_exp_group",
        default="",
        help="Not used by the script. Just used to allow for logging some "
        "info to organize experiments in Weights&Biases ",
    )

    return parser
