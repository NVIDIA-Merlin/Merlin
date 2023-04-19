import logging
import os
import time
from datetime import datetime

import tensorflow as tf

import wandb


class WandbLogger:
    def __init__(self, project, entity, config, logging_path=None):
        self.project = project
        self.entity = entity
        self.config = config
        self.logging_path = logging_path

    def setup(self):
        wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            dir=self.logging_path,
        )

    def log(self, metrics):
        wandb.log(metrics)

    def teardown(self):
        wandb.finish()


class ExamplesPerSecondCallback(tf.keras.callbacks.Callback):
    """ExamplesPerSecond callback.
    This callback records the average_examples_per_sec and
    current_examples_per_sec during training.
    """

    def __init__(self, batch_size, every_n_steps=1, log_as_print=True, log_to_wandb=True):
        self.log_as_print = log_as_print
        self.log_to_wandb = log_to_wandb
        self._batch_size = batch_size
        self._every_n_steps = every_n_steps
        super(ExamplesPerSecondCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self._first_batch = True
        self._epoch_steps = 0
        # self._train_start_time = time.time()
        # self._last_recorded_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        # Discards the first batch, as it is used to compile the
        # graph and affects the average
        if self._first_batch:
            self._epoch_steps = 0
            self._first_batch = False
            self._epoch_start_time = time.time()
            self._last_recorded_time = time.time()
            return

        """Log the examples_per_sec metric every_n_steps."""
        self._epoch_steps += 1
        current_time = time.time()

        if self._epoch_steps % self._every_n_steps == 0:
            average_examples_per_sec = self._batch_size * (
                self._epoch_steps / (current_time - self._epoch_start_time)
            )
            current_examples_per_sec = self._batch_size * (
                self._every_n_steps / (current_time - self._last_recorded_time)
            )

            if self.log_as_print:
                logging.info(
                    f"[Examples/sec - Epoch step: {self._epoch_steps}]"
                    f" current: {current_examples_per_sec:.2f}, avg: {average_examples_per_sec:.2f}"
                )

            if self.log_to_wandb:
                wandb.log(
                    {
                        "current_examples_per_sec": current_examples_per_sec,
                        "average_examples_per_sec": average_examples_per_sec,
                    }
                )

            self._last_recorded_time = current_time  # Update last_recorded_time


def get_callbacks(args):
    callbacks = [
        ExamplesPerSecondCallback(
            args.train_batch_size,
            every_n_steps=args.metrics_log_frequency,
            log_to_wandb=args.log_to_wandb,
        )
    ]

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

    if args.log_to_wandb:
        wandb_callback = wandb.keras.WandbCallback(
            log_batch_frequency=args.metrics_log_frequency,
            save_model=False,
            save_graph=False,
        )
        callbacks.append(wandb_callback)

    return callbacks
