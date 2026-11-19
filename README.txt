Pancreatic tumor segmentation

Code for comparing three 3D vision transformer models on pancreatic tumor
segmentation from portal-venous CT: UNETR, Swin UNETR and SegFormer3D.
The models are implemented from scratch in PyTorch under src/ptseg/models.

Dataset
-------
The dataset is not included in the repository. Download and prepare it with:

    python scripts/download_data.py
    python scripts/prepare_data.py

Prepared volumes and dataset.json are written to datasets/prepared.

Install
-------
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Train
-----
    python scripts/train.py -c configs/unetr.yml
    python scripts/train.py -c configs/swin_unetr.yml
    python scripts/train.py -c configs/segformer3d.yml

Each run writes checkpoints and metrics under outputs/runs/<run_name>.

Evaluate
--------
    python scripts/evaluate.py --run-dir outputs/runs/<run_name>

This reports Dice, IoU, precision, recall and HD95 per class (in mm), plus a
tumor-size breakdown (small / medium / large) saved to tumor_size_analysis.json.

Compare models
--------------
    python scripts/compare_models.py

Builds one table with segmentation metrics gathered from the runs and the cost
of each model (parameters, GFLOPs, forward time). If several models have been
evaluated, it also prints a paired Wilcoxon test on tumor Dice.

Use --train to train and evaluate every model before building the table.

Benchmark cost only
-------------------
    python scripts/benchmark.py

Tests
-----
    python tests/test_models.py
    python tests/test_config.py
    python tests/test_metrics.py

Layout
------
    src/ptseg     package (models, data, training, inference, metrics)
    scripts       command line entry points
    configs       YAML configuration
    tests         checks
