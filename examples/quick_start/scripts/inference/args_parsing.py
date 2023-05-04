import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Inference script for deploying ranking models with NVTabular on TIS"
    )

    # Inputs / Outputs
    parser.add_argument(
        "--nvt_workflow_path",
        default="./results/",
        help="Path to saved nvtabular workflow. ",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        help="If provided, loads a model saved by --save_model_path "
        "instead of initializing the parameters randomly",
    )
    parser.add_argument(
        "--ensemble_export_path",
        default="./models",
        help="Path for exporting the model artifacts to "
        "load them on Triton IS afterwards.",
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

    return parser


def parse_arguments():
    parser = build_arg_parser()
    args = parser.parse_args()

    return args
