# Lab 2 - Model training and experiment tracking with MLflow

## Question 1
After running `uv add mlflow torch torchvision scikit-learn`, `pyproject.toml` gained four new
dependencies: `mlflow`, `scikit-learn`, `torch`, and `torchvision`. It also gained a custom
package index section (`[[tool.uv.index]]` named `pytorch-cpu`, pointing at
`https://download.pytorch.org/whl/cpu`) plus a `[tool.uv.sources]` block routing `torch` and
`torchvision` to that index. This makes uv resolve the CPU-only PyTorch wheels instead of the
default CUDA build, avoiding a multi-GB download on a machine with no NVIDIA GPU. `uv.lock` was
regenerated to match, pinning the exact resolved versions and hashes of all new dependencies
and their transitive dependencies.

## Question 2
`--backend-store-uri` tells mlflow where to store run **metadata**: experiment/run records,
parameters, metrics, and tags. Here it's a local SQLite database (`sqlite:///mlflow.db`).
`--default-artifact-root` tells mlflow where to store **artifacts**: the actual files produced
by a run, such as the saved model, plots, or other output files (here, the local `./mlruns`
folder). The distinction is: metadata is small, structured, queryable data the UI reads to
build charts and tables; artifacts are the (often large) binary/file outputs of a run, stored
as files on disk rather than database rows.

## Question 3
`mlflow.db` and `mlruns/` shouldn't be tracked by git because they are local, machine-generated
outputs that change on every run — committing a SQLite file and a folder of binary artifacts to
git would bloat the repo's history with constantly-churning, non-source content. They shouldn't
be tracked by dvc either, because dvc is meant for versioning reproducible **data assets**
(inputs/outputs of the pipeline that need to be shared and reproduced across machines) — not
transient experiment-tracking logs that are naturally regenerated fresh by whoever runs the
training.

## Question 4
The first call to `mlflow.set_experiment("food11")` with a name that doesn't exist yet
automatically **creates** a new experiment with that name. Checking the mlflow UI afterward
shows "food11" listed alongside "Default" in the experiments sidebar.

## Question 5
`mlflow.log_param` records a fixed value that is set before training starts and doesn't change
during the run (e.g. learning rate, batch size, model architecture). `mlflow.log_metric` records
a value that evolves over the course of training (e.g. loss, accuracy). `log_metric` takes a
`step` argument because metrics are meant to be plotted as a time series across training
(e.g. per epoch) — mlflow needs to know at which point in training each value was recorded.
`log_param` has no such notion of "over time": a param is set once and stays constant for the
whole run, so there's nothing to plot against.

## Question 6
Opening the successful run (`redolent-gnat-579`) in the mlflow UI: the **Overview** tab shows
the 4 logged params (`dataset`, `epochs`, `lr`, `batch_size`) and the 4 logged metrics
(`train_loss`, `val_loss`, `val_accuracy`, `test_accuracy`) with their final values. The
**Model metrics** tab shows each metric as a chart plotted across the 5 epochs (steps 0–4),
letting you see the training curve rather than just the final number. The **Artifacts** tab
shows the logged model as a set of files: `MLmodel` (the model descriptor), a `data/` folder
(the actual model weights), plus environment/reproducibility files (`conda.yaml`,
`python_env.yaml`, `requirements.txt`) and the input example files
(`input_example.json`, `serving_input_example.json`) used to trace the model graph.

The model artifact physically lives on disk under
`./mlruns/1/35ecd2ab1e22419e83aa1ddee48254c5/artifacts/`, since `--default-artifact-root
./mlruns` was set when starting the server (experiment ID `1` and this run's ID make up the
rest of the path).

## Question 7
Comparing the four runs:

| lr | batch_size | val_accuracy (final epoch) | test_accuracy |
|---|---|---|---|
| 0.01 | 32 | 0.175 | 0.162 |
| 0.001 | 32 | 0.536 | 0.547 |
| 0.0001 | 32 | 0.775 | 0.809 |
| 0.001 | 64 | 0.605 | 0.642 |

`lr=0.0001` gave the best `val_accuracy` (0.775). Higher is **not** always better: `lr=0.01`
performed by far the worst (0.175) — a learning rate that's too high causes unstable, noisy
training (val_loss stayed high and jumped around instead of steadily decreasing) because the
optimizer takes steps too large to converge smoothly. Within the range tested, lower learning
rates gave better and smoother convergence.

## Question 8
Looking at `lr`, `batch_size`, and `val_accuracy` together in the parallel coordinates plot:
`val_accuracy` correlates strongly and inversely with `lr` — the lowest learning rate
(0.0001) produced the highest accuracy, and the highest learning rate (0.01) produced the
lowest, with `lr=0.001` runs landing in between. `batch_size` had a smaller but still visible
effect: at the same `lr=0.001`, increasing `batch_size` from 32 to 64 improved `val_accuracy`
(0.536 → 0.605), possibly because larger batches produced more stable gradient estimates during
training. Overall, `lr` had a much bigger impact on the outcome than `batch_size` did in this
comparison.

## Question 9
Sorting by `val_accuracy` descending, the best run is **bold-pug-293**
(`lr=0.0001, batch_size=32, epochs=5`), with `val_accuracy=0.775` and `test_accuracy=0.809`.

Run ID: `b7060166233c48ca99f28e85827260ff`