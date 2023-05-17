import gc
import logging
import os
from functools import reduce
from typing import Optional

import nvtabular as nvt
from merlin.core.dispatch import HAS_GPU
from merlin.schema import Tags
from nvtabular import ops as nvt_ops

from args_parsing import parse_arguments


def filter_by_freq(df_to_filter, df_for_stats, column, min_freq, max_freq=None):
    # Frequencies of each value in the column.
    freq = df_for_stats[column].value_counts()

    cond = freq == freq  # placeholder condition
    if min_freq is not None:
        cond = cond & (freq >= min_freq)
    if max_freq is not None:
        cond = cond & (freq <= max_freq)
    # Select frequent values. Value is in the index.
    frequent_values = freq[cond].reset_index()["index"].to_frame(column)
    # Return only rows with value frequency above threshold.
    return df_to_filter.merge(frequent_values, on=column, how="inner")


class PreprocessingRunner:
    def __init__(self, args):
        self.args = args

        self.gpu = HAS_GPU
        if self.gpu:
            import dask_cudf

            self.df_lib = dask_cudf
        else:
            import pandas

            self.df_lib = pandas

        self.dask_cluster_client = None

    def read_data(self, path):
        args = self.args
        logging.info(f"Reading data: {args.data_path}")
        if args.input_data_format in ["csv", "tsv"]:
            ddf = self.df_lib.read_csv(
                path, sep=args.csv_sep, na_values=args.csv_na_values
            )
        elif args.input_data_format == "parquet":
            ddf = self.df_lib.read_parquet(path)
        else:
            raise ValueError(f"Invalid input data format: {args.input_data_format}")

        logging.info("First lines...")
        logging.info(ddf.head())

        logging.info(f"Number of rows: {len(ddf)}")
        if args.filter_query:
            logging.info(f"Filtering rows using filter {args.filter_query}")
            ddf = ddf.query(args.filter_query)
            logging.info(f"Number of rows after filtering: {len(ddf)}")

        return ddf

    def cast_dtypes(self, ddf):
        logging.info("Converting dtypes")
        args = self.args
        columns = set(ddf.columns)
        if args.to_int32:
            ddf[args.to_int32] = ddf[
                list(set(args.to_int32).intersection(columns))
            ].astype("int32")
        if args.to_int16:
            ddf[args.to_int16] = ddf[
                list(set(args.to_int16).intersection(columns))
            ].astype("int16")
        if args.to_int8:
            ddf[args.to_int8] = ddf[
                list(set(args.to_int8).intersection(columns))
            ].astype("int8")
        if args.to_float32:
            ddf[args.to_float32] = ddf[
                list(set(args.to_float32).intersection(columns))
            ].astype("float32")
        return ddf

    def filter_by_user_item_freq(self, ddf):
        logging.info("Filtering rows with min/max user/item frequency")
        args = self.args

        filtered_ddf = ddf
        if (
            args.min_user_freq
            or args.max_user_freq
            or args.min_item_freq
            or args.max_item_freq
        ):
            print("Before filtering: ", len(filtered_ddf))
            for r in range(args.num_max_rounds_filtering):
                print(f"Round #{r+1}")
                if args.min_user_freq or args.max_user_freq:
                    filtered_ddf = filter_by_freq(
                        df_to_filter=filtered_ddf,
                        df_for_stats=filtered_ddf,
                        column=args.user_id_feature,
                        min_freq=args.min_user_freq,
                        max_freq=args.max_user_freq,
                    )
                    users_count = len(filtered_ddf)
                    print("After filtering users: ", users_count)
                if args.min_item_freq or args.max_item_freq:
                    filtered_ddf = filter_by_freq(
                        df_to_filter=filtered_ddf,
                        df_for_stats=filtered_ddf,
                        column=args.item_id_feature,
                        min_freq=args.min_item_freq,
                        max_freq=args.max_item_freq,
                    )
                    items_count = len(filtered_ddf)
                    print("After filtering items: ", items_count)

        return filtered_ddf

    def split_datasets(self, df):
        args = self.args
        if args.dataset_split_strategy == "random":
            logging.info(
                f"Splitting dataset into train and eval using strategy "
                f"'{args.dataset_split_strategy}'"
            )
            if self.gpu:
                # Converts dask_cudf to cudf DataFrame to split data
                df = df.compute()
            df = df.sample(frac=1.0).reset_index(drop=True)
            split_index = int(len(df) * args.random_split_eval_perc)

            train_df = df[:-split_index]
            eval_df = df[-split_index:]

            if self.gpu:
                train_df = self.df_lib.from_cudf(train_df, args.output_num_partitions)
                eval_df = self.df_lib.from_cudf(eval_df, args.output_num_partitions)

            return train_df, eval_df

        elif args.dataset_split_strategy == "random_by_user":
            if self.gpu:
                # Converts dask_cudf to cudf DataFrame to split data
                df = df.compute()
            df = df.sample(frac=1.0).reset_index(drop=True)

            # Getting number of examples per user
            users_count_df = (
                df.groupby(args.user_id_feature).size().to_frame("user_count")
            )
            df = df.merge(
                users_count_df, left_on=args.user_id_feature, right_index=True
            )

            # Assigning to each user example a percentage value according to the number
            # of available examples. For example, if the user has 20 events, each example
            # receive a cumulative percentage of 0.05: (0.05, 0.10, 0.15, ..., 0.95, 1.00)
            df["dummy"] = 1
            df["per_user_example_perc"] = (
                df.groupby(args.user_id_feature)["dummy"].cumsum() / df["user_count"]
            )
            df.drop(["dummy"], axis=1, inplace=True)
            # Using the percentage to split train and eval sets
            train_df = df[df["per_user_example_perc"] > args.random_split_eval_perc]
            eval_df = df[df["per_user_example_perc"] <= args.random_split_eval_perc]
            train_df.drop(["per_user_example_perc"], axis=1, inplace=True)
            eval_df.drop(["per_user_example_perc"], axis=1, inplace=True)

            if self.gpu:
                train_df = self.df_lib.from_cudf(train_df, args.output_num_partitions)
                eval_df = self.df_lib.from_cudf(eval_df, args.output_num_partitions)

            return train_df, eval_df

        elif args.dataset_split_strategy == "temporal":
            train_df = df[
                df[args.timestamp_feature] < args.dataset_split_temporal_timestamp
            ]
            eval_df = df[
                df[args.timestamp_feature] >= args.dataset_split_temporal_timestamp
            ]

            return train_df, eval_df

        else:
            raise ValueError(
                f"Invalid sampling strategy: {args.dataset_split_strategy}"
            )

    def generate_nvt_workflow_features(self):
        logging.info("Generating NVTabular workflow  for preprocessing features")
        args = self.args
        feats = dict()

        for col in args.control_features:
            feats[col] = [col]
        for col in args.categorical_features:
            feats[col] = [col] >> nvt_ops.Categorify(
                freq_threshold=args.categ_min_freq_capping
            )
        for col in args.continuous_features:
            feats[col] = [col]
            if args.continuous_features_fillna is not None:
                if args.continuous_features_fillna.lower() == "median":
                    feats[col] = feats[col] >> nvt_ops.FillMedian()
                else:
                    feats[col] = feats[col] >> nvt_ops.FillMissing(
                        args.continuous_features_fillna
                    )
                feats[col] = feats[col] >> nvt_ops.Normalize()

        for col in args.user_features:
            feats[col] = feats[col] >> nvt_ops.TagAsUserFeatures()
        for col in args.item_features:
            feats[col] = feats[col] >> nvt_ops.TagAsItemFeatures()

        if args.user_id_feature:
            feats[args.user_id_feature] = (
                feats[args.user_id_feature] >> nvt_ops.TagAsUserID()
            )

        if args.item_id_feature:
            feats[args.item_id_feature] = (
                feats[args.item_id_feature] >> nvt_ops.TagAsItemID()
            )

        if args.timestamp_feature:
            feats[args.timestamp_feature] = [args.timestamp_feature] >> nvt_ops.AddTags(
                [Tags.TIME]
            )

        if args.session_id_feature:
            feats[args.session_id_feature] = [
                args.session_id_feature
            ] >> nvt_ops.AddTags([Tags.SESSION_ID, Tags.SESSION, Tags.ID])

        # Combining all features
        outputs = reduce(lambda x, y: x + y, list(feats.values()))

        workflow = nvt.Workflow(outputs, client=self.dask_cluster_client)
        return workflow

    def generate_nvt_workflow_targets(self, client=None):
        logging.info("Generating NVTabular workflow for preprocessing targets")
        args = self.args
        feats = dict()

        for col in args.binary_classif_targets:
            feats[col] = [col] >> nvt_ops.AddTags(
                [Tags.BINARY_CLASSIFICATION, Tags.TARGET]
            )
        for col in args.regression_targets:
            feats[col] = [col] >> nvt_ops.AddTags(
                [Tags.REGRESSION, Tags.TARGET, Tags.BINARY]
            )

        # Combining all features
        outputs = reduce(lambda x, y: x + y, list(feats.values()))

        workflow = nvt.Workflow(outputs, client=self.dask_cluster_client)
        return workflow

    def persist_intermediate(self, ddf, folder):
        path = os.path.join(self.args.output_path, folder)
        logging.info(f"Persisting intermediate results to {path}")
        ddf.to_parquet(path)
        del ddf
        gc.collect()
        ddf = self.df_lib.read_parquet(path)
        return ddf

    def setup_dask_cuda_cluster(
        self,
        visible_devices: str = None,
        device_spill_frac: float = 0.7,
        dask_work_dir: Optional[str] = None,
    ):
        """Starts a Dask CUDA Cluster, so that multiple
        GPUs (memory and compute) can be used for the preprocessing.

        Parameters
        ----------
        visible_devices : str
            Comma separated list of visible GPU devices (e.g. "0,1")
        device_spill_frac : float
            Spill GPU-Worker memory to host at this limit.
        dask_work_dir : str
            Local path to use as dask work dir
        Returns
        -------
        Client
            Dask-distributed client
        """
        from dask.distributed import Client
        from dask_cuda import LocalCUDACluster
        from merlin.core.utils import device_mem_size

        capacity = device_mem_size(kind="total")  # Get device memory capacity
        # Reduce if spilling fails to prevent
        # device memory errors.
        cluster = None  # (Optional) Specify existing scheduler port
        if cluster is None:
            cluster = LocalCUDACluster(
                CUDA_VISIBLE_DEVICES="0,1",  # visible_devices,
                local_directory=dask_work_dir,
                device_memory_limit=capacity * device_spill_frac,
            )

        # Create the distributed client
        self.dask_cluster_client = Client(cluster)

    def run(self):
        args = self.args

        logging.info(f"Running device: {'GPU' if self.gpu else 'CPU'}")

        if self.gpu and args.enable_dask_cuda_cluster:
            logging.info("Setting up Dask CUDA Cluster")
            self.setup_dask_cuda_cluster(
                visible_devices=args.dask_cuda_visible_gpu_devices,
                device_spill_frac=args.dask_cuda_gpu_device_spill_frac,
                dask_work_dir=args.output_path,
            )

        ddf = self.read_data(args.data_path)
        ddf = self.cast_dtypes(ddf)
        ddf = self.filter_by_user_item_freq(ddf)

        if args.persist_intermediate_files:
            ddf = self.persist_intermediate(ddf, "_cache/01/")

        if args.eval_data_path:
            eval_ddf = self.read_data(args.eval_data_path)
            eval_ddf = self.cast_dtypes(eval_ddf)

        if args.predict_data_path:
            test_ddf = self.read_data(args.predict_data_path)
            test_ddf = self.cast_dtypes(test_ddf)

        if args.dataset_split_strategy:
            if args.eval_data_path:
                raise ValueError(
                    "You cannot provide both --eval_data_path and --dataset_split_strategy"
                )

            ddf, eval_ddf = self.split_datasets(ddf)

            if args.persist_intermediate_files:
                ddf = self.persist_intermediate(ddf, "_cache/02/train/")
                eval_ddf = self.persist_intermediate(eval_ddf, "_cache/02/eval/")

        nvt_workflow_features = self.generate_nvt_workflow_features()
        nvt_workflow_targets = self.generate_nvt_workflow_targets()

        logging.info("Fitting/transforming the preprocessing on train set")

        output_dataset_path = args.output_path

        train_dataset = nvt.Dataset(ddf, cpu=not self.gpu)
        # Processing features and targets in separate workflows, because
        # targets might not be available for test/predict_dataset
        train_dataset_features = nvt_workflow_features.fit_transform(train_dataset)
        train_dataset_targets = nvt_workflow_targets.fit_transform(train_dataset)
        train_dataset_preproc = nvt.Dataset(
            self.df_lib.concat(
                [train_dataset_features.to_ddf(), train_dataset_targets.to_ddf()],
                axis=1,
            ),
            schema=train_dataset_features.schema + train_dataset_targets.schema,
            cpu=not self.gpu,
        )

        output_train_dataset_path = os.path.join(output_dataset_path, "train")
        logging.info(f"Fitting and transforming train set: {output_train_dataset_path}")
        train_dataset_preproc.to_parquet(
            output_train_dataset_path,
            output_files=args.output_num_partitions,
        )

        if args.eval_data_path or args.dataset_split_strategy:
            eval_dataset = nvt.Dataset(eval_ddf, cpu=not self.gpu)
            # Processing features and targets in separate workflows, because
            # targets might not be available for test/predict_dataset
            eval_dataset_features = nvt_workflow_features.transform(eval_dataset)
            eval_dataset_targets = nvt_workflow_targets.transform(eval_dataset)
            eval_dataset_preproc = nvt.Dataset(
                self.df_lib.concat(
                    [eval_dataset_features.to_ddf(), eval_dataset_targets.to_ddf()],
                    axis=1,
                ),
                schema=eval_dataset_features.schema + eval_dataset_targets.schema,
                cpu=not self.gpu,
            )

            output_eval_dataset_path = os.path.join(output_dataset_path, "eval")
            logging.info(f"Transforming eval set: {output_eval_dataset_path}")

            eval_dataset_preproc.to_parquet(
                output_eval_dataset_path,
                output_files=args.output_num_partitions,
            )

        if args.predict_data_path:
            predict_dataset = nvt.Dataset(test_ddf, cpu=not self.gpu)
            new_predict_dataset = nvt_workflow_features.transform(predict_dataset)

            output_predict_dataset_path = os.path.join(output_dataset_path, "predict")
            logging.info(f"Transforming predict set: {output_predict_dataset_path}")

            new_predict_dataset.to_parquet(
                output_predict_dataset_path,
                output_files=args.output_num_partitions,
            )
        nvt_save_path = os.path.join(output_dataset_path, "workflow")
        logging.info(f"Saving nvtabular workflow to: {nvt_save_path}")
        nvt_workflow_features.save(nvt_save_path)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_arguments()

    runner = PreprocessingRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
