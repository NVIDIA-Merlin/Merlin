import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocessing script for ranking models using NVTabular and dask-cudf on GPUs"
    )

    # Inputs / Outputs
    parser.add_argument("--data_path", default="", help="Path to the data")
    parser.add_argument(
        "--eval_data_path",
        default=None,
        help="Path to eval data, if data was already split"
        "Must have the same schema as train data (in --data_path).",
    )
    parser.add_argument(
        "--predict_data_path",
        default=None,
        help="Path to data to be preprocessed for prediction."
        "This data is expected to have the same input features as train data but not targets, "
        "as this data is used for prediction.",
    )
    parser.add_argument(
        "--input_data_format",
        default="csv",
        choices=["csv", "tsv", "parquet"],
        help="Input data format",
    )

    parser.add_argument(
        "--csv_sep",
        default=",",
        help="Character separator for CSV files."
        "Default is ','. You can use 'tab' for tabular separated data, or "
        "--input_data_format tsv",
    )
    parser.add_argument(
        "--csv_na_values",
        default=None,
        help="String in the original data that should be " "replaced by NULL",
    )

    parser.add_argument(
        "--output_path",
        default="./results/",
        help="Output path where the preprocessed files "
        "and nvtabular workflow will be saved"
        "Default is ./results/",
    )
    parser.add_argument(
        "--output_num_partitions",
        default=10,
        type=int,
        help="Number of partitions "
        "that result in this number of output files"
        "Default is 10.",
    )
    parser.add_argument(
        "--persist_intermediate_files",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to persist/cache the intermediate preprocessing files. "
        "Enabling this might be necessary for larger datasets.",
    )

    parser.add_argument(
        "--control_features",
        default="",
        help="Columns (comma-separated) that should be kept as is in the output files. "
        "For example, --control_features=session_id,timestamp",
    )
    parser.add_argument(
        "--categorical_features",
        default="",
        help="Columns (comma-sep) with categorical/discrete features "
        "that will encoded/categorified to contiguous ids in the preprocessing. "
        "These tags are tagged as 'categorical' in the schema, "
        "so that Merlin Models can automatically create "
        "embedding tables for them.",
    )
    parser.add_argument(
        "--continuous_features",
        default="",
        help="Columns (comma-sep) with continuous features that will be standardized "
        "and tagged in the schema as 'continuous', so that the Merlin Models can "
        "represent and combine them with embedding properly.",
    )

    parser.add_argument(
        "--continuous_features_fillna",
        default=None,
        help="Replaces NULL values with this float. "
        "You can also set it with 'median' for filling nulls with the median value.",
    )

    parser.add_argument(
        "--user_features",
        default="",
        help="Columns (comma-sep) that should be tagged in the schema  as user features. "
        "This information might be useful for modeling later.",
    )
    parser.add_argument(
        "--item_features",
        default="",
        help="Columns (comma-sep) that should be tagged in the schema as item features. "
        "This information might be useful for modeling later, "
        "for example, for in-batch sampling if your data contains only positive examples.",
    )
    parser.add_argument(
        "--binary_classif_targets",
        default="",
        help="Columns (comma-sep) that should be tagged in the schema as binary target. "
        "Merlin Models will create a binary classification head for each of these targets.",
    )
    parser.add_argument(
        "--regression_targets",
        default="",
        help="Columns (comma-sep) that should be tagged in the schema as binary target. "
        "Merlin Models will create a regression head for each of these targets.",
    )

    parser.add_argument(
        "--user_id_feature",
        default="",
        help="Column that contains the user id feature, for tagging in the schema. "
        "This information is used in the preprocessing "
        "if you set --min_user_freq or --max_user_freq",
    )
    parser.add_argument(
        "--item_id_feature",
        default="",
        help="Column that contains the item id feature, for tagging in the schema. "
        "This information is used in the preprocessing "
        "if you set --min_item_freq or --max_item_freq",
    )
    parser.add_argument(
        "--timestamp_feature",
        default="",
        help="Column containing a timestamp or date feature. "
        "The basic preprocessing doesn't extracts date and time features for it. "
        "It is just tagged as 'timestamp' in "
        "the schema and used for splitting train / eval data "
        "if --dataset_split_strategy=temporal is used.",
    )
    parser.add_argument(
        "--session_id_feature",
        default="",
        help="This is just for tagging this feature.",
    )

    # parser.add_argument("--groupby_feature", default="", help="")

    parser.add_argument(
        "--to_int32", default="", help="Cast these columns (comma-sep) to int32."
    )
    parser.add_argument(
        "--to_int16",
        default="",
        help="Cast these columns (comma-sep) to int16, to save some memory.",
    )
    parser.add_argument(
        "--to_int8",
        default="",
        help="Cast these columns (comma-sep) to int32, to save some memory.",
    )
    parser.add_argument(
        "--to_float32", default="", help="Cast these columns (comma-sep) to float32"
    )

    parser.add_argument(
        "--categ_min_freq_capping",
        default=0,
        type=int,
        help="Value used for min frequency capping. If greater than 0, "
        "all categorical values which are less frequent than this "
        "threshold will be mapped to the null value encoded id.",
    )

    parser.add_argument(
        "--min_user_freq",
        default=None,
        type=int,
        help="Users with frequency lower than this value "
        "are removed from the dataset (before data splitting).",
    )
    parser.add_argument(
        "--max_user_freq",
        default=None,
        type=int,
        help="Users with frequency higher than this value "
        "are removed from the dataset (before data splitting).",
    )
    parser.add_argument(
        "--min_item_freq",
        default=None,
        type=int,
        help="Items with frequency lower than this value "
        "are removed from the dataset (before data splitting).",
    )
    parser.add_argument(
        "--max_item_freq",
        default=None,
        type=int,
        help="Items with frequency higher than this value "
        "are removed from the dataset (before data splitting).",
    )
    parser.add_argument(
        "--num_max_rounds_filtering",
        default=5,
        type=int,
        help="Max number of rounds interleaving users and items frequency filtering. "
        "If a small number of rounds is chosen, some low-frequent users or items "
        "might be kept in the dataset. Default is 5",
    )

    parser.add_argument(
        "--filter_query",
        default=None,
        type=str,
        help="A filter query condition compatible with dask-cudf `DataFrame.query()`",
    )

    parser.add_argument(
        "--dataset_split_strategy",
        default=None,
        type=str,
        choices=["random", "random_by_user", "temporal"],
        help="If None, no data split is performed. "
        "If 'random', samples are assigned randomly to eval set according to "
        "--random_split_eval_perc. "
        "If 'random_by_user', users will have examples in both train and eval set, "
        "according to the proportion "
        "specified in --random_split_eval_perc. "
        "If 'temporal', the --timestamp_feature with --dataset_split_temporal_timestamp "
        "to split eval set based on time.",
    )

    parser.add_argument(
        "--random_split_eval_perc",
        default=None,
        type=float,
        help="Percentage of examples to be assigned to eval set. "
        "It is used with --dataset_split_strategy 'random' and 'random_by_user'",
    )
    parser.add_argument(
        "--dataset_split_temporal_timestamp",
        default=None,
        type=int,
        help="Used when --dataset_split_strategy 'temporal'. "
        "It assigns for eval set all examples where the --timestamp_feature >= value",
    )

    parser.add_argument(
        "--enable_dask_cuda_cluster",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Initializes a LocalCUDACluster for multi-GPU preprocessing.",
    )

    parser.add_argument(
        "--dask_cuda_visible_gpu_devices",
        default=None,
        type=str,
        help="Ids of GPU devices that should be used "
        "for preprocessing, if any. For example: --visible_gpu_devices=0,1. "
        "Default is None, for using all GPUs",
    )
    parser.add_argument(
        "--dask_cuda_gpu_device_spill_frac",
        default=0.7,
        type=float,
        help="Percentage of GPU memory used at which "
        "LocalCUDACluster should spill memory to CPU, before raising "
        "out-of-memory errors. Default is 0.7",
    )

    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_list_arg(v):
    if v is None or v == "":
        return []
    return v.split(",")


def parse_arguments():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Parsing list args
    args.control_features = parse_list_arg(args.control_features)
    args.categorical_features = parse_list_arg(args.categorical_features)
    args.continuous_features = parse_list_arg(args.continuous_features)

    args.binary_classif_targets = parse_list_arg(args.binary_classif_targets)
    args.regression_targets = parse_list_arg(args.regression_targets)

    args.user_features = parse_list_arg(args.user_features)
    args.item_features = parse_list_arg(args.item_features)
    args.to_int32 = parse_list_arg(args.to_int32)
    args.to_int16 = parse_list_arg(args.to_int16)
    args.to_int8 = parse_list_arg(args.to_int8)
    args.to_float32 = parse_list_arg(args.to_float32)

    if args.filter_query:
        args.filter_query = args.filter_query.replace('"', "")

    if args.csv_sep.lower() == "tab" or args.input_data_format == "tsv":
        args.csv_sep = "\t"

    return args
