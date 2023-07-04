import merlin.models.tf as mm
from tensorflow.keras import regularizers

from .args_parsing import MtlArgsPrefix, Task


def get_task_sample_weights(
    target, binary_output, tasks_pos_class_weights, tasks_space
):
    sample_weight_block = None
    if binary_output:
        sample_weight_block = mm.ColumnBasedSampleWeight(
            weight_column_name=target,
            binary_class_weights=(1.0, tasks_pos_class_weights[target]),
        )

    if tasks_space[target]:
        sample_space_block = mm.ColumnBasedSampleWeight(
            weight_column_name=tasks_space[target]
        )

        if sample_weight_block:
            sample_weight_block = mm.SequentialBlock(
                # Cascaded sample weights
                sample_weight_block,
                sample_space_block,
            )

    return sample_weight_block


def get_mtl_loss_weights(args, targets):
    flattened_targets = [y for x in targets.values() for y in x]

    loss_weights = {
        target: float(args[f"{MtlArgsPrefix.LOSS_WEIGHT_ARG_PREFIX.value}{target}"])
        if f"{MtlArgsPrefix.LOSS_WEIGHT_ARG_PREFIX.value}{target}" in args
        else 1.0
        for target in flattened_targets
    }

    mtl_loss_weights = {}
    if Task.BINARY_CLASSIFICATION.value in targets:
        mtl_loss_weights.update(
            {
                f"{target}/binary_output": loss_weights[target]
                for target in targets[Task.BINARY_CLASSIFICATION.value]
            }
        )
    if Task.REGRESSION.value in targets:
        mtl_loss_weights.update(
            {
                f"{target}/regression_output": loss_weights[target]
                if target in loss_weights
                else 1.0
                for target in targets[Task.REGRESSION.value]
            }
        )

    return mtl_loss_weights


def get_mtl_positive_class_weights(targets, args):
    pos_class_weights = {}
    if Task.BINARY_CLASSIFICATION.value in targets:
        pos_class_weights = {
            target: float(
                args[f"{MtlArgsPrefix.POS_CLASS_WEIGHT_ARG_PREFIX.value}{target}"]
            )
            if f"{MtlArgsPrefix.POS_CLASS_WEIGHT_ARG_PREFIX.value}{target}" in args
            else 1.0
            for target in targets[Task.BINARY_CLASSIFICATION.value]
        }
    return pos_class_weights


def get_mtl_prediction_tasks(targets, args):
    task_block = None
    if args.use_task_towers:
        task_block = mm.MLPBlock(
            args.tower_layers,
            activation=args.activation,
            kernel_initializer=args.mlp_init,
            dropout=args.dropout,
            kernel_regularizer=regularizers.l2(args.l2_reg),
            bias_regularizer=regularizers.l2(args.l2_reg),
        )

    tasks_pos_class_weights = get_mtl_positive_class_weights(targets, args)

    if args.tasks_sample_space:
        if len(args.tasks) != len(args.tasks_sample_space):
            raise ValueError(
                "If --tasks_sample_space is provided, the list of tasks sample "
                "(separated by ',') need to match the length of the list "
                "of --tasks ('all' is not allowed in this case. "
                "If the sample space of a target is the whole dataset "
                "then you can use empty string ('') for that task. "
                "For example: --tasks=click,like --tasks_sample_space=,click"
            )
        tasks_space = dict(zip(args.tasks, args.tasks_sample_space))
    else:
        tasks_space = {t: None for t in args.tasks}

    prediction_tasks = []
    if Task.BINARY_CLASSIFICATION.value in targets:
        prediction_tasks.extend(
            [
                mm.BinaryOutput(
                    target,
                    post=get_task_sample_weights(
                        target, True, tasks_pos_class_weights, tasks_space
                    ),
                    # Cloning task tower
                    pre=task_block.from_config(task_block.get_config())
                    if task_block
                    else None,
                )
                for target in targets[Task.BINARY_CLASSIFICATION.value]
            ]
        )

    if Task.REGRESSION.value in targets:
        prediction_tasks.extend(
            [
                mm.RegressionOutput(
                    target,
                    post=get_task_sample_weights(
                        target, False, tasks_pos_class_weights, tasks_space
                    ),
                    # Cloning task tower
                    pre=task_block.from_config(task_block.get_config())
                    if task_block
                    else None,
                )
                for target in targets[Task.REGRESSION.value]
            ]
        )

    prediction_tasks = mm.ParallelBlock(*prediction_tasks)
    return prediction_tasks
