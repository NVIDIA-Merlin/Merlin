from functools import partial

import merlin.models.tf as mm
from merlin.models.tf import Model
from merlin.models.utils.schema_utils import infer_embedding_dim
from merlin.schema.tags import Tags
from tensorflow.keras import regularizers


def get_model(schema, prediction_tasks, args):
    if args.model == "mlp":
        model = get_mlp_model(schema, args, prediction_tasks)
    elif args.model == "dcn":
        model = get_dcn_model(schema, args, prediction_tasks)
    elif args.model == "dlrm":
        model = get_dlrm_model(schema, args, prediction_tasks)
    elif args.model == "mlp":
        model = get_mlp_model(schema, args, prediction_tasks)
    elif args.model == "wide_n_deep":
        model = get_wide_and_deep_model(schema, args, prediction_tasks)
    elif args.model == "deepfm":
        model = get_deepfm_model(schema, args, prediction_tasks)
    # Multi-Task Learning specific models
    elif args.model == "mmoe":
        model = get_mmoe_model(schema, args, prediction_tasks)
    elif args.model == "cgc":
        model = get_cgc_model(schema, args, prediction_tasks)
    elif args.model == "ple":
        model = get_ple_model(schema, args, prediction_tasks)
    else:
        raise ValueError(f"Invalid option for --model: {args.model}")

    return model


# mlp model
def get_mlp_model(schema, args, prediction_tasks):
    cat_schema = schema.select_by_tag(Tags.CATEGORICAL)

    input_block = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            cat_schema,
            embeddings_regularizer=regularizers.l2(args.embeddings_l2_reg),
            infer_dim_fn=partial(
                infer_embedding_dim, multiplier=args.embedding_sizes_multiplier,
            ),
        ),
        aggregation="concat",
    )

    mlp = mm.MLPBlock(
        args.mlp_layers,
        activation=args.activation,
        kernel_initializer=args.mlp_init,
        dropout=args.dropout,
        kernel_regularizer=regularizers.l2(args.l2_reg),
        bias_regularizer=regularizers.l2(args.l2_reg),
    )

    model = Model(input_block, mlp, prediction_tasks)

    return model


# dcn model
def get_dcn_model(schema, args, prediction_tasks):
    # Keeping only categorical features and removing the user id (keeping only seq features)
    # schema_selected = schema.select_by_tag(Tags.CATEGORICAL).remove_by_tag(Tags.USER_ID)

    input_block = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            schema.select_by_tag(Tags.CATEGORICAL),
            embeddings_regularizer=regularizers.l2(args.embeddings_l2_reg),
            infer_dim_fn=partial(
                infer_embedding_dim, multiplier=args.embedding_sizes_multiplier,
            ),
        ),
        aggregation="concat",
    )

    model = mm.DCNModel(
        schema,
        input_block=input_block,
        depth=args.dcn_interacted_layer_num,
        deep_block=mm.MLPBlock(
            args.mlp_layers,
            activation=args.activation,
            kernel_initializer=args.mlp_init,
            dropout=args.dropout,
            kernel_regularizer=regularizers.l2(args.l2_reg),
            bias_regularizer=regularizers.l2(args.l2_reg),
        ),
        prediction_tasks=prediction_tasks,
    )

    return model


def get_dlrm_model(schema, args, prediction_tasks):
    # Keeping only categorical features and removing the user id (keeping only seq features)
    # schema_selected = schema.select_by_tag(Tags.CATEGORICAL).remove_by_tag(Tags.USER_ID)

    model = mm.DLRMModel(
        schema,
        embeddings=mm.Embeddings(
            schema.select_by_tag(Tags.CATEGORICAL),
            dim=args.embeddings_dim,
            embeddings_regularizer=regularizers.l2(args.embeddings_l2_reg),
        ),
        bottom_block=mm.MLPBlock(
            [args.embeddings_dim],
            activation=args.activation,
            kernel_initializer=args.mlp_init,
            dropout=args.dropout,
            kernel_regularizer=regularizers.l2(args.l2_reg),
            bias_regularizer=regularizers.l2(args.l2_reg),
        ),
        top_block=mm.MLPBlock(
            args.mlp_layers,
            activation=args.activation,
            kernel_initializer=args.mlp_init,
            dropout=args.dropout,
            kernel_regularizer=regularizers.l2(args.l2_reg),
            bias_regularizer=regularizers.l2(args.l2_reg),
        ),
        prediction_tasks=prediction_tasks,
    )

    return model


def get_deepfm_model(schema, args, prediction_tasks):
    cat_schema = schema.select_by_tag(Tags.CATEGORICAL)
    cat_schema_onehot = cat_schema.remove_by_tag(Tags.SEQUENCE).remove_by_tag(
        Tags.USER_ID
    )
    cat_schema_multihot = cat_schema.select_by_tag(Tags.SEQUENCE)

    input_block = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            cat_schema,
            dim=args.embeddings_dim,
            embeddings_regularizer=regularizers.l2(args.embeddings_l2_reg),
        ),
        aggregation=None,
    )

    wide_inputs_block = {
        "categorical_ohe": mm.SequentialBlock(
            mm.Filter(cat_schema_onehot),
            mm.CategoryEncoding(cat_schema_onehot, sparse=True, output_mode="one_hot"),
        )
    }

    if len(cat_schema_multihot) > 0:
        wide_inputs_block["categorical_mhe"] = mm.SequentialBlock(
            mm.Filter(cat_schema_multihot),
            mm.ListToDense(max_seq_length=args.multihot_max_seq_length),
            mm.CategoryEncoding(
                cat_schema_multihot, sparse=True, output_mode="multi_hot"
            ),
        )

    model = mm.DeepFMModel(
        cat_schema,
        deep_block=mm.MLPBlock(
            args.mlp_layers,
            activation=args.activation,
            kernel_initializer=args.mlp_init,
            dropout=args.dropout,
            kernel_regularizer=regularizers.l2(args.l2_reg),
            bias_regularizer=regularizers.l2(args.l2_reg),
        ),
        input_block=input_block,
        wide_input_block=mm.ParallelBlock(wide_inputs_block, aggregation="concat"),
        prediction_tasks=prediction_tasks,
    )

    return model


def get_wide_and_deep_model(schema, args, prediction_tasks):
    cat_schema = schema.select_by_tag(Tags.CATEGORICAL)
    cat_schema_multihot = cat_schema.select_by_tag(Tags.SEQUENCE).remove_by_tag(
        Tags.USER_ID
    )
    cat_schema_onehot = cat_schema.remove_by_tag(Tags.SEQUENCE).remove_by_tag(
        Tags.USER_ID
    )

    deep_embedding = mm.Embeddings(
        cat_schema,
        embeddings_regularizer=regularizers.l2(args.embeddings_l2_reg),
        infer_dim_fn=partial(
            infer_embedding_dim, multiplier=args.embedding_sizes_multiplier,
        ),
    )

    ignore_combinations = None
    if args.wnd_ignore_combinations:
        ignore_combinations = [x.split(":") for x in args.wnd_ignore_combinations]

    wide_preprocess = [
        # One-hot features
        mm.SequentialBlock(
            mm.Filter(cat_schema_onehot),
            mm.CategoryEncoding(cat_schema_onehot, sparse=True, output_mode="one_hot"),
        ),
        # 2nd level feature interactions of multi-hot features
        mm.SequentialBlock(
            mm.Filter(cat_schema.remove_by_tag(Tags.USER_ID)),
            mm.ListToDense(max_seq_length=args.multihot_max_seq_length),
            mm.HashedCrossAll(
                cat_schema.remove_by_tag(Tags.USER_ID),
                num_bins=args.wnd_hashed_cross_num_bins,
                max_level=2,
                output_mode="multi_hot",
                sparse=True,
                ignore_combinations=ignore_combinations,
            ),
        ),
    ]
    if len(cat_schema_multihot) > 0:
        wide_preprocess.append(
            mm.SequentialBlock(
                mm.Filter(cat_schema_multihot),
                mm.ListToDense(max_seq_length=args.multihot_max_seq_length),
                mm.CategoryEncoding(
                    cat_schema_multihot, sparse=True, output_mode="multi_hot"
                ),
            )
        )

    model = mm.WideAndDeepModel(
        schema,
        deep_input_block=mm.InputBlockV2(schema=cat_schema, categorical=deep_embedding),
        deep_block=mm.MLPBlock(
            args.mlp_layers,
            dropout=args.dropout,
            kernel_regularizer=regularizers.l2(args.l2_reg),
            bias_regularizer=regularizers.l2(args.l2_reg),
        ),
        wide_schema=schema.remove_by_tag(Tags.TARGET),
        deep_regularizer=regularizers.l2(args.l2_reg),
        wide_regularizer=regularizers.l2(args.wnd_wide_l2_reg),
        wide_dropout=args.dropout,
        deep_dropout=args.dropout,
        wide_preprocess=mm.ParallelBlock(wide_preprocess, aggregation="concat",),
        prediction_tasks=prediction_tasks,
    )

    return model


# mmoe model
def get_mmoe_model(schema, args, prediction_tasks):
    expert_block = mm.MLPBlock(
        args.expert_mlp_layers,
        activation=args.activation,
        kernel_initializer=args.mlp_init,
        no_activation_last_layer=False,
        dropout=args.dropout,
        kernel_regularizer=regularizers.l2(args.l2_reg),
        bias_regularizer=regularizers.l2(args.l2_reg),
    )

    input_block = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            schema.select_by_tag(Tags.CATEGORICAL),
            embeddings_regularizer=regularizers.l2(args.embeddings_l2_reg),
            infer_dim_fn=partial(
                infer_embedding_dim, multiplier=args.embedding_sizes_multiplier,
            ),
        ),
        aggregation="concat",
    )

    mmoe_kwargs = {}
    if args.gate_dim > 0:
        mmoe_kwargs["gate_block"] = mm.MLPBlock([args.gate_dim])
    mmoe = mm.MMOEBlock(
        prediction_tasks,
        expert_block=expert_block,
        num_experts=args.mmoe_num_mlp_experts,
        gate_softmax_temperature=args.mtl_gates_softmax_temperature,
        enable_gate_weights_metrics=True,
        **mmoe_kwargs,
    )

    return mm.Model(input_block, mmoe, prediction_tasks)


def get_cgc_model(schema, args, prediction_tasks):
    expert_block_mlp = mm.MLPBlock(
        args.expert_mlp_layers,
        activation=args.activation,
        kernel_initializer=args.mlp_init,
        dropout=args.dropout,
        kernel_regularizer=regularizers.l2(args.l2_reg),
        bias_regularizer=regularizers.l2(args.l2_reg),
    )

    input_block = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            schema.select_by_tag(Tags.CATEGORICAL),
            embeddings_regularizer=regularizers.l2(args.embeddings_l2_reg),
            infer_dim_fn=partial(
                infer_embedding_dim, multiplier=args.embedding_sizes_multiplier,
            ),
        ),
        aggregation="concat",
    )

    cgc_kwargs = {}
    if args.gate_dim > 0:
        cgc_kwargs["gate_block"] = mm.MLPBlock([args.gate_dim])
    cgc = mm.CGCBlock(
        prediction_tasks,
        expert_block=expert_block_mlp,
        num_task_experts=args.cgc_num_task_experts,
        num_shared_experts=args.cgc_num_shared_experts,
        gate_softmax_temperature=args.mtl_gates_softmax_temperature,
        enable_gate_weights_metrics=True,
        **cgc_kwargs,
    )

    return Model(input_block, cgc, prediction_tasks)


def get_ple_model(schema, args, prediction_tasks):
    expert_block_mlp = mm.MLPBlock(
        args.expert_mlp_layers,
        activation=args.activation,
        kernel_initializer=args.mlp_init,
        dropout=args.dropout,
        kernel_regularizer=regularizers.l2(args.l2_reg),
        bias_regularizer=regularizers.l2(args.l2_reg),
    )

    input_block = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            schema.select_by_tag(Tags.CATEGORICAL),
            embeddings_regularizer=regularizers.l2(args.embeddings_l2_reg),
            infer_dim_fn=partial(
                infer_embedding_dim, multiplier=args.embedding_sizes_multiplier,
            ),
        ),
        aggregation="concat",
    )

    ple_kwargs = {}
    if args.gate_dim > 0:
        ple_kwargs["gate_block"] = mm.MLPBlock([args.gate_dim])
    cgc = mm.PLEBlock(
        num_layers=args.ple_num_layers,
        outputs=prediction_tasks,
        expert_block=expert_block_mlp,
        num_task_experts=args.cgc_num_task_experts,
        num_shared_experts=args.cgc_num_shared_experts,
        gate_softmax_temperature=args.mtl_gates_softmax_temperature,
        enable_gate_weights_metrics=True,
        **ple_kwargs,
    )

    return Model(input_block, cgc, prediction_tasks)
