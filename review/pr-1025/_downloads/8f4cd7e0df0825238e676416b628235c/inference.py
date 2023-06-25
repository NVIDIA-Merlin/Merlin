import json
import logging
import os
import shutil

import tensorflow as tf
import merlin.models.tf as mm
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow
from nvtabular.workflow import Workflow

from args_parsing import parse_arguments


def main(args):
    logging.basicConfig(level=logging.INFO)

    workflow_stored_path = os.path.join(args.nvt_workflow_path, "workflow")
    logging.info(f"Loading nvt workflow from: {workflow_stored_path}")
    workflow = Workflow.load(workflow_stored_path)
    logging.info("Printing out workflow input schema")
    logging.info(workflow.input_schema.column_names)

    # load the tensorflow model
    tf_model_path = args.load_model_path
    logging.info(f"Loading saved ranking model from: {tf_model_path}")
    model = tf.keras.models.load_model(tf_model_path)

    logging.info("Creating ensemble graph")
    # create ensemble graph
    serving_operators = (
        workflow.input_schema.column_names
        >> TransformWorkflow(workflow)
        >> PredictTensorflow(model)
    )

    ensemble = Ensemble(serving_operators, workflow.input_schema)

    ens_model_path = args.ensemble_export_path

    logging.info("Removing the existing ensemble model path and creating a new folder")
    if os.path.isdir(ens_model_path):
        shutil.rmtree(ens_model_path)
    os.mkdir(ens_model_path)

    logging.info(f"Exporting model artifacts to: {ens_model_path}")
    ens_conf, node_confs = ensemble.export(ens_model_path)
    outputs = ensemble.graph.output_schema.column_names

    logging.info("Saving model output names to disk as a json file")
    output_targets_path = os.path.join("./", "outputs.json")
    json_object = json.dumps(outputs, indent=4)
    with open(output_targets_path, "w") as outfile:
        outfile.write(json_object)

    logging.info("Finished exporting models and config files")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
