import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Inference script for deploying ranking models with NVTabular on Triton"
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
        help="Loads a model from its saved path. ",
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
