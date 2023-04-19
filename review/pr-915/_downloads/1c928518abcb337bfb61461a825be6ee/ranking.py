import logging
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from args_parsing import Task, parse_arguments
from mtl import get_mtl_loss_weights, get_mtl_prediction_tasks
from ranking_models import get_model
from run_logging import WandbLogger, get_callbacks
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Mean

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.tf.transforms.negative_sampling import InBatchNegatives
from merlin.schema.tags import Tags


def get_datasets(args):
    train_ds = Dataset(os.path.join(args.train_path, "*.parquet"), part_size="500MB")
    eval_ds = Dataset(os.path.join(args.eval_path, "*.parquet"), part_size="500MB")

    return train_ds, eval_ds


class RankingTrainEvalRunner:
    logger = None
    train_ds = None
    eval_ds = None
    train_loader = None
    eval_loader = None
    args = None

    def __init__(self, logger, train_ds, eval_ds, args):
        self.args = args
        self.logger = logger
        self.train_ds = train_ds
        self.eval_ds = eval_ds

        self.schema, eval_schema, self.targets = self.filter_schema_with_selected_targets(
            self.train_ds.schema, self.eval_ds.schema
        )
        self.set_dataloaders(self.schema, eval_schema)

    def get_targets(self, schema):
        tasks = self.args.tasks
        if "all" in tasks:
            tasks = schema.select_by_tag(Tags.TARGET).column_names

        targets_schema = schema.select_by_name(tasks)
        targets = dict()

        binary_classif_targets = targets_schema.select_by_tag(
            Tags.BINARY_CLASSIFICATION
        ).column_names
        if len(binary_classif_targets) > 0:
            targets[Task.BINARY_CLASSIFICATION] = binary_classif_targets

        regression_targets = targets_schema.select_by_tag(Tags.REGRESSION).column_names
        if len(regression_targets) > 0:
            targets[Task.REGRESSION] = regression_targets

        return targets

    def filter_schema_with_selected_targets(self, train_schema, eval_schema):
        targets = self.get_targets(train_schema)

        if "all" not in self.args.tasks:
            flattened_targets = [y for x in targets.values() for y in x]
            # Removing targets not used from schema
            targets_to_remove = list(
                set(train_schema.select_by_tag(Tags.TARGET).column_names).difference(
                    set(flattened_targets)
                )
            )
            train_schema = train_schema.excluding_by_name(targets_to_remove)
            eval_schema = eval_schema.excluding_by_name(targets_to_remove)

        return train_schema, eval_schema, targets

    def set_dataloaders(self, train_schema, eval_schema):
        args = self.args
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

    def get_metrics(self):
        metrics = dict()
        if Task.BINARY_CLASSIFICATION in self.targets:
            metrics.update(
                {
                    f"{target}/binary_output": [
                        tf.keras.metrics.AUC(name="auc", curve="ROC", num_thresholds=int(1e5)),
                        tf.keras.metrics.AUC(name="prauc", curve="PR", num_thresholds=int(1e5)),
                        LogLossMetric(name="logloss"),
                    ]
                    for target in self.targets[Task.BINARY_CLASSIFICATION]
                }
            )

        if Task.REGRESSION in self.targets:
            metrics.update(
                {f"{target}/regression_output": "rmse" for target in self.targets[Task.REGRESSION]}
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
            opt = tf.keras.optimizers.Adagrad(
                learning_rate=lerning_rate,
            )
        else:
            raise ValueError("Invalid optimizer")

        return opt

    def train_eval_stl(self):
        if Task.BINARY_CLASSIFICATION in self.targets:
            prediction_task = mm.BinaryOutput(self.targets[Task.BINARY_CLASSIFICATION][0])
        elif Task.REGRESSION in self.targets:
            prediction_task = mm.RegressionOutput(self.targets[Task.REGRESSION][0])
        else:
            raise ValueError(f"Unrecognized task: {self.targets}")

        model = get_model(self.schema, prediction_task, self.args)

        metrics = self.get_metrics()
        model.compile(
            self.get_optimizer(),
            run_eagerly=False,
            metrics=metrics,
        )

        callbacks = get_callbacks(self.args)
        class_weights = {0: 1.0, 1: self.args.stl_positive_class_weight}

        fit_kwargs = {}
        if not self.args.predict:
            fit_kwargs = {
                "validation_data": self.eval_loader,
                "validation_steps": self.args.validation_steps,
            }

        logging.info("Starting to train the model")
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

        if self.args.predict:
            self.save_predictions(model, self.eval_loader.dataset)

        else:
            logging.info("Starting the evaluation of the model")

            eval_metrics = model.evaluate(
                self.eval_loader,
                batch_size=self.args.eval_batch_size,
                return_dict=True,
                callbacks=callbacks,
            )

            logging.info(f"EVALUATION METRICS: {eval_metrics}")
            self.log_final_metrics(eval_metrics)

        return model

    def train_eval_mtl(self):
        args = self.args

        prediction_tasks = get_mtl_prediction_tasks(self.targets, self.args)

        model = get_model(self.schema, prediction_tasks, self.args)

        loss_weights = get_mtl_loss_weights(args, self.targets)

        metrics = self.get_metrics()
        model.compile(
            self.get_optimizer(),
            run_eagerly=False,
            metrics=metrics,
            loss_weights=loss_weights,
        )
        callbacks = get_callbacks(self.args)

        logging.info(f"MODEL: {model}")

        fit_kwargs = {}
        if not self.args.predict:
            fit_kwargs = {
                "validation_data": self.eval_loader,
                "validation_steps": self.args.validation_steps,
            }

        logging.info("Starting to train the model (fit())")
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

        if self.args.predict:
            self.save_predictions(model, self.eval_loader.dataset)

        else:
            logging.info("Starting the evaluation the model (evaluate())")

            eval_metrics = model.evaluate(
                self.eval_loader,
                batch_size=args.eval_batch_size,
                return_dict=True,
                callbacks=callbacks,
            )

            auc_metric_results = {
                k.split("/")[0]: v for k, v in eval_metrics.items() if "binary_output/auc" in k
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

        return model

    def save_predictions(self, model, dataset):
        logging.info("Starting the batch predict of the evaluation set")

        predictions_ds = model.batch_predict(
            dataset,
            batch_size=self.args.eval_batch_size,
        )
        predictions_ddf = predictions_ds.to_ddf()

        if self.args.predict_keep_cols:
            cols = set(predictions_ddf.columns)
            pred_cols = sorted(
                list(cols.difference(set(self.eval_loader.dataset.to_ddf().columns)))
            )
            # Keeping only selected features and all targets
            predictions_ddf = predictions_ddf[self.args.predict_keep_cols + pred_cols]

        output_path = self.args.predict_output_path

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

    def log_final_metrics(self, metrics_results):
        if self.logger:
            metrics_results = {f"{k}-final": v for k, v in metrics_results.items()}
            self.logger.log(metrics_results)

    def run(self):
        if self.logger:
            self.logger.setup()

        tf.keras.utils.set_random_seed(self.args.random_seed)

        try:
            logging.info(f"TARGETS: {self.targets}")

            if len(self.targets) == 1 and len(list(self.targets.values())[0]) == 1:
                # Single target = Single-Task Learning
                model = self.train_eval_stl()
            else:
                # Multiple targets = Multi-Task Learning
                model = self.train_eval_mtl()

            logging.info("Finished training / evaluation / prediction")

            if self.args.save_trained_model_path:
                logging.info(f"Saving model to {self.args.save_trained_model_path}")
                model.save(self.args.save_trained_model_path)

            logging.info("Script successfully finished")

        finally:
            if self.logger:
                self.logger.teardown()


def main():
    logging.basicConfig(level=logging.DEBUG)

    args = parse_arguments()

    os.makedirs(args.output_path, exist_ok=True)

    logger = None
    if args.log_to_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            logging_path=args.output_path,
        )

    train_ds, eval_ds = get_datasets(args)

    runner = RankingTrainEvalRunner(logger, train_ds, eval_ds, args)
    runner.run()


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class LogLossMetric(Mean):
    """Log loss metric.
    Keras offers the log loss (a.k.a. binary cross entropy), but it
    may be affected by sample weights, which you might not
    want for the metric calculation).
    This is the corresponding metric, that is useful to evaluate
    the performance of binary classification tasks.

    Parameters
    ----------
    name : str, optional
        Name of the metric, by default "logloss"
    from_logits : bool, optional
        Whether the metric should expect the likelihood (e.g. sigmoid function in output)
        or logits (False). By default False
    dtype
        Dtype of metric output, by default None
    """

    def __init__(self, name="logloss", from_logits=False, dtype=None):
        self.from_logits = from_logits
        super().__init__(name=name, dtype=dtype)

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ):
        result = binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        return super().update_state(result, sample_weight=sample_weight)

    def get_config(self):
        config = super().get_config()
        config["from_logits"] = self.from_logits
        return config


if __name__ == "__main__":
    main()
