# mykrml_26148130

A small, reusable Python package of machine-learning utility functions built for the
36120 Advanced Machine Learning Application course (UTS). It bundles common data-preparation,
feature-engineering, and model-scoring helpers used across the assignment notebooks so that
the same, tested logic can be shared between experiments.

## Installation

The package is published to TestPyPI:

```bash
pip install mykrml_26148130 --extra-index-url https://test.pypi.org/simple/
```

With `uv` (recommended, keeps a shared project's `pyproject.toml` unchanged):

```bash
uv pip install mykrml_26148130 --extra-index-url https://test.pypi.org/simple/ --index-strategy unsafe-best-match
```

## Modules and functions

### `mykrml_26148130.data.sets`

| Function | Purpose |
|---|---|
| `pop_target(df, target_col)` | Split a dataframe into features and target (returns `features, target`). |
| `save_sets(X_train, y_train, X_val, y_val, X_test, y_test, path)` | Persist any provided sets as `.joblib` files. |
| `load_sets(path)` | Load previously saved sets; any set not on disk is returned as `None`. |
| `subset_x_y(target, features, start_index, end_index)` | Slice features and target to the same row range. |
| `split_sets_by_time(features, target, test_ratio)` | Chronological train/validation/test split (most recent rows become test). |
| `split_sets_random(features, target, test_ratio, random_state)` | Reproducible random train/validation/test split. |

### `mykrml_26148130.features.dates`

| Function | Purpose |
|---|---|
| `convert_to_date(df, cols)` | Convert the given columns to `datetime` (returns a new dataframe). |

### `mykrml_26148130.models.performance`

| Function | Purpose |
|---|---|
| `print_regressor_scores(y_preds, y_actuals, set_name)` | Print RMSE and MAE for a regression model. |
| `print_classifier_scores(y_preds, y_actuals, set_name)` | Print Accuracy and F1 (binary or weighted, chosen automatically). |
| `print_auprc(y_scores, y_actuals, set_name)` | Print AUPRC (average precision) from predicted probabilities/scores. |
| `assess_classifier_set(model, features, target, set_name)` | Predict with a trained model and print classifier scores. |
| `fit_assess_classifier(model, X_train, y_train, X_val, y_val)` | Fit a classifier, print train/validation scores, return the model. |

## Usage examples

```python
from mykrml_26148130.data.sets import pop_target, split_sets_random, save_sets, load_sets
from mykrml_26148130.features.dates import convert_to_date
from mykrml_26148130.models.performance import print_classifier_scores, print_auprc

# Separate the target
features, target = pop_target(df=my_dataframe, target_col="label")

# Reproducible train/validation/test split
X_train, y_train, X_val, y_val, X_test, y_test = split_sets_random(
    features, target, test_ratio=0.2, random_state=42
)

# Persist and reload processed sets
save_sets(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, path="../data/processed/")
X_train, y_train, X_val, y_val, X_test, y_test = load_sets(path="../data/processed/")

# Convert date columns
df = convert_to_date(df, cols=["date_of_birth"])

# Score a classifier
print_classifier_scores(y_preds=preds, y_actuals=y_val, set_name="Validation")
print_auprc(y_scores=probabilities, y_actuals=y_val, set_name="Validation")
```

## Development

```bash
uv sync            # install dependencies
uv run pytest      # run the unit tests
```

The test suite (`tests/test_mykrml_26148130.py`) covers every function across all three modules.

## License

MIT