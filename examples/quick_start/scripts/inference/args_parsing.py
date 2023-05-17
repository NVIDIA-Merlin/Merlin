import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Inference script for deploying ranking models with NVTabular on Triton"
    )

    # Inputs / Outputs
    parser.add_argument(
        "--nvt_workflow_path",
        default=None,
        help="Loads the nvtabular workflow saved in the preprocessing step. ",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        help="Loads a model saved by --save_model_path in the ranking step. ",
    )
    parser.add_argument(
        "--ensemble_export_path",
        default="./models",
        help="Path for exporting the model artifacts to "
        "load them on Triton inference server afterwards.",
    )

    return parser


def parse_arguments():
    parser = build_arg_parser()
    args = parser.parse_args()

    return args
