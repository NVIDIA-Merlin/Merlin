import json
import logging
import os
from datetime import datetime

import merlin.models.tf as mm
import numpy as np
import tensorflow as tf
from args_parsing import Task, parse_arguments
from merlin.io.dataset import Dataset
from merlin.models.tf.logging.callbacks import ExamplesPerSecondCallback, WandbLogger
from merlin.models.tf.transforms.negative_sampling import InBatchNegatives
from merlin.schema.tags import Tags

from args_parsing import Task, parse_arguments
from mtl import get_mtl_loss_weights, get_mtl_prediction_tasks
from ranking_models import get_model



def get_datasets(args):
    train_ds = (
        Dataset(os.path.join(args.train_data_path, "*.parquet"), part_size="500MB")
        if args.train_data_path
        else None
    )
    eval_ds = (
        Dataset(os.path.join(args.eval_data_path, "*.parquet"), part_size="500MB")
        if args.eval_data_path
        else None
    )
    predict_ds = (
        Dataset(os.path.join(args.predict_data_path, "*.parquet"), part_size="500MB")
        if args.predict_data_path
        else None
    )

    return train_ds, eval_ds, predict_ds


class RankingTrainEvalRunner:
    logger = None
    train_ds = None
    eval_ds = None
    train_loader = None
    eval_loader = None
    predict_loader = None
    args = None

    def __init__(self, logger, train_ds, eval_ds, predict_ds, args):
        self.args = args
        self.logger = logger
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.predict_ds = predict_ds

        (
            self.schema,
            eval_schema,
            self.targets,
        ) = self.filter_schema_with_selected_targets()
        self.set_dataloaders(self.schema, eval_schema)

    def get_targets(self, schema):
        tasks = self.args.tasks
        if "all" in tasks:
            tasks = schema.select_by_tag(Tags.TARGET).column_names

        targets_schema = schema.select_by_name(tasks)

        if set(tasks) != set(targets_schema.column_names):
            raise ValueError(
                "Some tasks were not found in the dataset schema: "
                f"{set(tasks).difference(set(targets_schema.column_names))}"
            )

        targets = dict()
        binary_classif_targets = targets_schema.select_by_tag(
            Tags.BINARY_CLASSIFICATION
        ).column_names
        if len(binary_classif_targets) > 0:
            targets[Task.BINARY_CLASSIFICATION.value] = binary_classif_targets

        regression_targets = targets_schema.select_by_tag(Tags.REGRESSION).column_names
        if len(regression_targets) > 0:
            targets[Task.REGRESSION.value] = regression_targets

        return targets

    def filter_schema_with_selected_targets(self):
        targets = None
        train_schema = None
        if self.train_ds:
            train_schema = self.train_ds.schema
            targets = self.get_targets(train_schema)

        eval_schema = None
        if self.eval_ds:
            eval_schema = self.eval_ds.schema
            targets = self.get_targets(eval_schema)

        if targets and "all" not in self.args.tasks:
            flattened_targets = [y for x in targets.values() for y in x]
            # Removing targets not used from schema
            targets_to_remove = list(
                set(
                    (train_schema or eval_schema)
                    .select_by_tag(Tags.TARGET)
                    .column_names
                ).difference(set(flattened_targets))
            )
            if train_schema:
                train_schema = train_schema.excluding_by_name(targets_to_remove)
            if eval_schema:
                eval_schema = eval_schema.excluding_by_name(targets_to_remove)

        schema = train_schema or eval_schema or self.predict_ds.schema
        return schema, eval_schema, targets

    def set_dataloaders(self, train_schema, eval_schema):
        args = self.args

        self.train_loader = None
        if self.train_ds:
            train_loader_kwargs = {}
            if self.args.in_batch_negatives_train:
                train_loader_kwargs["transform"] = InBatchNegatives(
                    train_schema, args.in_batch_negatives_train
                )
            self.train_loader = mm.Loader(
                self.train_ds,
                batch_size=args.train_batch_size,
                schema=train_schema,
                **train_loader_kwargs,
            )

        self.eval_loader = None
        if self.eval_ds:
            eval_loader_kwargs = {}
            if args.in_batch_negatives_eval:
                eval_loader_kwargs["transform"] = InBatchNegatives(
                    eval_schema, args.in_batch_negatives_eval
                )

            self.eval_loader = mm.Loader(
                self.eval_ds,
                batch_size=args.eval_batch_size,
                schema=eval_schema,
                **eval_loader_kwargs,
            )

        self.predict_loader = None
        if self.predict_ds:
            self.predict_loader = mm.Loader(
                self.predict_ds,
                batch_size=args.eval_batch_size,
            )

    def get_metrics(self):
        metrics = dict()
        if Task.BINARY_CLASSIFICATION.value in self.targets:
            metrics.update(
                {
                    f"{target}/binary_output": [
                        tf.keras.metrics.AUC(
                            name="auc", curve="ROC", num_thresholds=int(1e5)
                        ),
                        tf.keras.metrics.AUC(
                            name="prauc", curve="PR", num_thresholds=int(1e5)
                        ),
                        mm.LogLossMetric(name="logloss"),
                    ]
                    for target in self.targets[Task.BINARY_CLASSIFICATION.value]
                }
            )

        if Task.REGRESSION.value in self.targets:
            metrics.update(
                {
                    f"{target}/regression_output": "rmse"
                    for target in self.targets[Task.REGRESSION.value]
                }
            )

        if len(metrics) == 1:
            return list(metrics.values())[0]
        else:
            return metrics

    def get_optimizer(self):
        lerning_rate = self.args.lr
        if self.args.lr_decay_rate:
            lerning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                self.args.lr,
                decay_steps=self.args.lr_decay_steps,
                decay_rate=self.args.lr_decay_rate,
                staircase=True,
            )

        if self.args.optimizer == "adam":
            opt = tf.keras.optimizers.Adam(
                learning_rate=lerning_rate,
            )
        elif self.args.optimizer == "adagrad":
            opt = tf.keras.optimizers.legacy.Adagrad(
                learning_rate=lerning_rate,
            )
        else:
            raise ValueError("Invalid optimizer")

        return opt

    def build_stl_model(self):
        if Task.BINARY_CLASSIFICATION.value in self.targets:
            prediction_task = mm.BinaryOutput(
                self.targets[Task.BINARY_CLASSIFICATION.value][0]
            )
        elif Task.REGRESSION in self.targets:
            prediction_task = mm.RegressionOutput(
                self.targets[Task.REGRESSION.value][0]
            )
        else:
            raise ValueError(f"Unrecognized task: {self.targets}")

        model = get_model(self.schema, prediction_task, self.args)
        return model

    def train_eval_stl(self, model):
        metrics = self.get_metrics()
        model.compile(
            self.get_optimizer(),
            run_eagerly=False,
            metrics=metrics,
        )

        callbacks = self.get_callbacks(self.args)
        class_weights = {0: 1.0, 1: self.args.stl_positive_class_weight}

        if self.train_loader:
            logging.info("Starting to train the model")

            fit_kwargs = {}
            if self.eval_loader:
                fit_kwargs = {
                    "validation_data": self.eval_loader,
                    "validation_steps": self.args.validation_steps,
                }

            model.fit(
                self.train_loader,
                epochs=self.args.epochs,
                batch_size=self.args.train_batch_size,
                steps_per_epoch=self.args.train_steps_per_epoch,
                shuffle=False,
                drop_last=False,
                callbacks=callbacks,
                train_metrics_steps=self.args.train_metrics_steps,
                class_weight=class_weights,
                **fit_kwargs,
            )

        if self.eval_loader:
            logging.info("Starting the evaluation of the model")

            eval_metrics = model.evaluate(
                self.eval_loader,
                batch_size=self.args.eval_batch_size,
                return_dict=True,
                callbacks=callbacks,
            )

            logging.info(f"EVALUATION METRICS: {eval_metrics}")
            self.log_final_metrics(eval_metrics)

        if self.predict_loader:
            self.save_predictions(model, self.predict_loader.dataset)

    def build_mtl_model(self):
        prediction_tasks = get_mtl_prediction_tasks(self.targets, self.args)
        model = get_model(self.schema, prediction_tasks, self.args)
        return model

    def train_eval_mtl(self, model):
        args = self.args

        loss_weights = get_mtl_loss_weights(args, self.targets)

        metrics = self.get_metrics()
        model.compile(
            self.get_optimizer(),
            run_eagerly=False,
            metrics=metrics,
            loss_weights=loss_weights,
        )
        callbacks = self.get_callbacks(self.args)

        if self.train_loader:
            logging.info("Starting to train the model (fit())")
            fit_kwargs = {}
            if self.eval_loader:
                fit_kwargs = {
                    "validation_data": self.eval_loader,
                    "validation_steps": self.args.validation_steps,
                }

            model.fit(
                self.train_loader,
                epochs=args.epochs,
                batch_size=args.train_batch_size,
                steps_per_epoch=args.train_steps_per_epoch,
                shuffle=False,
                drop_last=False,
                callbacks=callbacks,
                train_metrics_steps=args.train_metrics_steps,
                **fit_kwargs,
            )

        if self.eval_loader:
            logging.info("Starting the evaluation the model (evaluate())")

            eval_metrics = model.evaluate(
                self.eval_loader,
                batch_size=args.eval_batch_size,
                return_dict=True,
                callbacks=callbacks,
            )

            auc_metric_results = {
                k.split("/")[0]: v
                for k, v in eval_metrics.items()
                if "binary_output/auc" in k
            }

            auc_metric_results = {f"{k}-auc": v for k, v in auc_metric_results.items()}

            avg_metrics = {
                "auc_avg": np.mean(list(auc_metric_results.values())),
            }

            all_metrics = {
                **avg_metrics,
                **auc_metric_results,
                **eval_metrics,
            }

            logging.info(f"EVALUATION METRICS: {all_metrics}")

            # log final metrics
            self.log_final_metrics(all_metrics)

        if self.predict_loader:
            logging.info("Starting to save predictions")
            self.save_predictions(model, self.predict_loader.dataset)

    def save_predictions(self, model, dataset):
        logging.info("Starting the batch predict of the evaluation set")

        predictions_ds = model.batch_predict(
            dataset,
            batch_size=self.args.eval_batch_size,
        )
        predictions_ddf = predictions_ds.to_ddf()

        if self.args.predict_output_keep_cols:
            cols = set(predictions_ddf.columns)
            pred_cols = sorted(list(cols.difference(set(dataset.to_ddf().columns))))
            # Keeping only selected features and all predictions
            predictions_ddf = predictions_ddf[
                self.args.predict_output_keep_cols + pred_cols
            ]

        if not self.args.predict_output_path:
            raise Exception(
                "You need to specify the path to save the predictions "
                "using --predict_output_path"
            )
        output_path = self.args.predict_output_path

        logging.info(f"Saving predictions to {output_path}")

        if self.args.predict_output_format == "parquet":
            predictions_ddf.to_parquet(output_path, write_index=False)
        elif self.args.predict_output_format in ["csv", "tsv"]:
            if self.args.predict_output_format == "csv":
                sep = ","
            elif self.args.predict_output_format == "tsv":
                sep = "\t"
            predictions_ddf.to_csv(output_path, single_file=True, index=False, sep=sep)
        else:
            raise ValueError(
                "Only supported formats for output prediction files"
                f" are parquet or csv, but got '{self.args.predict_output_format}'"
            )

        logging.info(f"Predictions saved to {output_path}")

    def log_final_metrics(self, metrics_results):
        if self.logger:
            metrics_results = {f"{k}-final": v for k, v in metrics_results.items()}
            self.logger.log(metrics_results)

    def run(self):
        if self.logger:
            self.logger.init()

        tf.keras.utils.set_random_seed(self.args.random_seed)

        try:
            logging.info(f"TARGETS: {self.targets}")

            model = None
            if self.args.load_model_path:
                model = self.load_model(self.args.load_model_path)

            # If a single-task learning model (if only --predict_data_path is
            # provided, as self.targets will not be available, it checks the
            # --tasks arg to discover if its STL or MTL)
            if len(self.targets) == 1 and len(list(self.targets.values())[0]) == 1:
                if not model:
                    model = self.build_stl_model()

                logging.info(f"MODEL: {model}")
                # Single target = Single-Task Learning
                self.train_eval_stl(model)
            else:
                if not model:
                    model = self.build_mtl_model()

                logging.info(f"MODEL: {model}")
                # Multiple targets = Multi-Task Learning
                self.train_eval_mtl(model)

            if self.args.save_model_path:
                logging.info("Saving the model")
                self.save_model(model, self.args.save_model_path)

            logging.info("Script finished successfully")

        finally:
            if self.logger:
                self.logger.teardown()

    def save_model(self, model, path):
        logging.info(f"Saving model to {path}")
        model.save(path)

        # Saving the model targets
        output_targets_path = os.path.join(path, "targets.json")
        logging.info(f"Saving model targets metadata to {output_targets_path}")
        json_object = json.dumps(self.targets, indent=4)
        with open(output_targets_path, "w") as outfile:
            outfile.write(json_object)

    def load_model(self, path):
        logging.info(f"Loading model from: {path}")
        model = mm.Model.load(path)

        # Loading the model targets
        output_targets_path = os.path.join(path, "targets.json")
        logging.info(f"Loading model targets metadata from: {output_targets_path}")
        with open(output_targets_path, "r") as outfile:
            self.targets = json.loads(outfile.read())

        return model

    def get_callbacks(self, args):
        callbacks = []

        if args.log_to_tensorboard:
            logdir = os.path.join(
                args.output_path, "tb_logs/", datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=logdir,
                update_freq=args.metrics_log_frequency,
                write_steps_per_second=True,
            )
            callbacks.append(tensorboard_callback)

        wandb_callback = None
        if args.log_to_wandb:
            wandb_callback = self.logger.get_callback(
                metrics_log_frequency=args.metrics_log_frequency,
                save_model=False,
                save_graph=False,
            )
            callbacks.append(wandb_callback)

        callbacks.append(
            [
                ExamplesPerSecondCallback(
                    args.train_batch_size,
                    every_n_steps=args.metrics_log_frequency,
                    logger=self.logger,
                    log_to_console=True,
                )
            ]
        )

        return callbacks


def main():
    logging.basicConfig(level=logging.DEBUG)

    args = parse_arguments()

    os.makedirs(args.output_path, exist_ok=True)

    logger = None
    if args.log_to_wandb:
        logger = WandbLogger(
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            config=args,
            logging_path=args.output_path,
        )
        logger.init()

    train_ds, eval_ds, predict_ds = get_datasets(args)

    runner = RankingTrainEvalRunner(logger, train_ds, eval_ds, predict_ds, args)
    runner.run()


if __name__ == "__main__":
    main()
