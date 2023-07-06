from asvdb import ASVDb, BenchmarkResult, utils
from testbook import testbook

from tests.conftest import REPO_ROOT, get_benchmark_info


@testbook(
    REPO_ROOT / "examples/Next-Item-Prediction-with-Transformers/tf/transformers-next-item-prediction.ipynb",
    timeout=720,
    execute=False,
)
def test_func(tb, tmpdir):
    tb.inject(
        f"""
        import os
        os.environ["INPUT_DATA_DIR"] = "/raid/data/booking"
        os.environ["OUTPUT_DATA_DIR"] = "{tmpdir}"
        os.environ["NUM_EPOCHS"] = '1'
        """
    )
    tb.cells.pop(6)
    tb.cells[
        15
    ].source = """
        def process_data():
            wf = Workflow(filtered_sessions)

            wf.fit_transform(train_set_dataset).to_parquet(
                os.path.join(OUTPUT_DATA_DIR, 'train_processed.parquet')
            )
            wf.transform(validation_set_dataset).to_parquet(
                os.path.join(OUTPUT_DATA_DIR, 'validation_processed.parquet')
            )

            wf.save(os.path.join(OUTPUT_DATA_DIR, 'workflow'))

        data_processing_runtime = timeit.timeit(process_data, number=1)
    """
    tb.cells[
        29
    ].source = """
        model.compile(run_eagerly=False, optimizer='adam', loss="categorical_crossentropy")

        def train_model():
            model.fit(
                Dataset(os.path.join(OUTPUT_DATA_DIR, 'train_processed.parquet')),
                batch_size=64,
                epochs=NUM_EPOCHS,
                pre=mm.SequenceMaskRandom(
                    schema=seq_schema,
                    target=target,
                    masking_prob=0.3,
                    transformer=transformer_block
                    )
                )

        training_runtime = timeit.timeit(train_model, number=1)
    """
    tb.execute_cell(list(range(0, 35)))
    data_processing_runtime = tb.ref("data_processing_runtime")
    training_runtime = tb.ref("training_runtime")
    ndcg_at_10 = tb.ref("metrics")["ndcg_at_10"]

    bResult1 = BenchmarkResult(
        funcName="",
        argNameValuePairs=[
            ("notebook_name", "usecases/transformers-next-item-prediction"),
            ("measurement", "data_processing_runtime"),
        ],
        result=data_processing_runtime,
    )
    bResult2 = BenchmarkResult(
        funcName="",
        argNameValuePairs=[
            ("notebook_name", "usecases/transformers-next-item-prediction"),
            ("measurement", "training_runtime"),
        ],
        result=training_runtime,
    )
    bResult3 = BenchmarkResult(
        funcName="",
        argNameValuePairs=[
            ("notebook_name", "usecases/transformers-next-item-prediction"),
            ("measurement", "ndcg_at_10"),
        ],
        result=ndcg_at_10,
    )

    bInfo = get_benchmark_info()
    (repo, branch) = utils.getRepoInfo()

    db = ASVDb(dbDir="s3://nvtab-bench-asvdb/models_metric_tracking", repo=repo, branches=[branch])
    db.addResult(bInfo, bResult1)
    db.addResult(bInfo, bResult2)
    db.addResult(bInfo, bResult3)
