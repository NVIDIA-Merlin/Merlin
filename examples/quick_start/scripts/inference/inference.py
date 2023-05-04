import logging
import os
import shutil

import tensorflow as tf
from merlin.schema.tags import Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow
from nvtabular.workflow import Workflow

from args_parsing import parse_arguments


def main(args):
    logging.basicConfig(level=logging.INFO)

    workflow_stored_path = os.path.join(args.nvt_workflow_path, "workflow")
    logging.info(f"loading nvt workflow from: {workflow_stored_path}")
    workflow = Workflow.load(workflow_stored_path)

    logging.info("removing target columns from workflow input schema")
    label_columns = workflow.output_schema.select_by_tag(Tags.TARGET).column_names
    workflow.remove_inputs(label_columns)
    logging.info("printing out workflow input schema")
    logging.info(workflow.input_schema)
    logging.info("printing out workflow output schema")
    logging.info(workflow.output_schema)

    # load the tensorflow model
    tf_model_path = args.load_model_path
    logging.info(f"loading saved ranking model from: {tf_model_path}")
    model = tf.keras.models.load_model(tf_model_path)

    # create ensemble graph
    serving_operators = (
        workflow.input_schema.column_names
        >> TransformWorkflow(workflow)
        >> PredictTensorflow(model)
    )

    ensemble = Ensemble(serving_operators, workflow.input_schema)

    ens_model_path = args.ensemble_export_path

    # Make sure we have a clean stats space
    if os.path.isdir(ens_model_path):
        shutil.rmtree(ens_model_path)
    os.mkdir(ens_model_path)

    # store the exported models and the config files.
    ens_conf, node_confs = ensemble.export(ens_model_path)

    logging.info("Finished exporting triton models and config files")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
