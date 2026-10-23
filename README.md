# From Examples to Generalization: An ML Classification Case Study

[![ML Pipeline Tests](https://github.com/snehasunill/ml-classification-workshop/actions/workflows/ml-tests.yml/badge.svg)](https://github.com/snehasunill/ml-classification-workshop/actions/workflows/ml-tests.yml)


A reproducible **supervised machine learning case study** and a companion, simplified educational demo. The technical study compares classifiers, tunes hyperparameters using cross-validation, diagnoses overfitting, and examines how incomplete training coverage affects a subgroup. The teaching demo translates the same concepts for a Grades 6–9 audience.

> **Scope:** All pet data are synthetic and deliberately simplistic. The results demonstrate methodology, not real animal recognition or a real-world fairness audit. This is a personal prototype, not an official GIRLsmarts4tech resource.

## What this project demonstrates

| ML concept | Concrete implementation | Why it matters |
|---|---|---|
| Baseline | Majority-class `DummyClassifier` | Establishes a reference beyond which model performance should improve |
| Preprocessing | `StandardScaler` inside `Pipeline` for logistic regression and k-NN | Fits scaling *inside each CV training fold*, preventing preprocessing leakage; distance-based k-NN is scale-sensitive |
| Model selection | Stratified 5-fold `GridSearchCV` over k-NN, logistic regression, and decision trees | Compares models on **validation folds**, not on the final test data |
| Held-out evaluation | 80/20 stratified split, class-specific F1, precision, recall, accuracy, and confusion matrix | Provides an estimate of generalization after model selection |
| Bias–variance / overfitting | Tree depth sweep with noisy training labels and separate train-vs-CV fold scores | Shows why increasing model complexity can hurt validation accuracy |
| Representation gaps | Train with vs. without small dogs, evaluate against the same held-out subgroup | Reveals how training coverage can change subgroup recall |
| Reproducibility | Fixed seeds, automated tests, JSON output, saved diagnostic figure | Makes findings inspectable rather than relying on screenshot claims |


## Experimental Results
### Model comparison using cross-validation

The dataset contains 900 synthetic observations, split into 720 training and 180 held-out test examples. Hyperparameter tuning and model selection use stratified 5-fold cross-validation on the training partition only.

| Model | Best mean CV dog-class F1 |
|---|---:|
| Majority-class baseline | 0.000 |
| Logistic regression | 0.935 |
| k-NN | **0.939** |
| Decision tree | 0.936 |

k-NN was selected based on the highest mean cross-validation F1 score, with `n_neighbors=11`.

### Final held-out test evaluation

Only the CV-selected benchmark model is evaluated on the held-out test set.

| Metric | Selected k-NN |
|---|---:|
| Accuracy | 93.3% |
| Dog-class precision | 100% |
| Dog-class recall | 86.7% |
| Dog-class F1 | 0.929 |
| Balanced accuracy | 93.3% |

The confusion matrix (rows = actual, columns = predicted; class order: cat, dog) is:

| | Predicted cat | Predicted dog |
|---|---:|---:|
| Actual cat | 90 | 0 |
| Actual dog | 12 | 78 |

Although overall accuracy is high, the classifier misses 12 of 90 dogs. This demonstrates why class-specific metrics are important alongside accuracy.


### Overfitting diagnostic

![Mean train vs validation accuracy across decision-tree depths](reports/tree_complexity.png)

With injected label noise in the separate complexity experiment, the unlimited-depth tree achieves **100% mean fold training accuracy**, versus **72.2% mean CV validation accuracy**. The gap is **27.8 percentage points**, evidence of poor generalization on this deliberately noisy setup. This diagnostic uses only the training split; it does **not** tune models on the final test set.

### Subgroup coverage experiment

On **33 held-out small dogs**, dog-class recall changes from **9.1%** (3/33) when small dogs are excluded from training to **66.7%** (22/33) when they are included. The evaluation group is identical for both conditions. Removing a subgroup also changes training-set size/class balance, so this is a *representation-gap demonstration*, not a controlled estimate of one isolated causal factor.


## Technical Interpretation & Discussion

### Model selection and uncertainty

k-NN achieved the highest mean CV F1 (0.939), but logistic regression (0.935) and decision trees (0.936) performed similarly. These small differences do not establish statistical superiority.

The selected model is the winner under the specified cross-validation procedure, not necessarily the best model across other datasets or random splits. Repeated cross-validation could help assess how stable the rankings are.

### Why feature scaling matters

k-NN classifies observations based on distances between feature vectors. Without scaling, a feature with a larger numerical range may dominate distance calculations.

StandardScaler is included inside a scikit-learn Pipeline, ensuring the scaler is fitted separately on each CV training fold. This prevents information from validation folds leaking into preprocessing.

### Overfitting and the bias-variance tradeoff

As decision-tree depth increases, training accuracy improves while validation accuracy eventually declines. In the noise-injected experiment, the unrestricted tree reaches 100% training accuracy but only 72.2% CV accuracy.

This growing generalization gap is evidence of overfitting: the tree increasingly fits training-specific patterns that do not generalize well.

### Training-data representation

Excluding small dogs from training reduces their held-out recall from 66.7% to 9.1%. This illustrates how incomplete training coverage can disproportionately affect a subgroup.

However, excluding small dogs also changes the training-set size and class distribution. Therefore, the experiment cannot isolate subgroup coverage as the sole causal explanation.

### Limitations

The dataset is synthetic and contains only two numerical features. The conclusions demonstrate ML methodology rather than real-world animal classification performance. Further analysis could include repeated CV, learning curves, uncertainty estimates, and controlled subgroup ablations.



## Run locally

Python 3.10+ recommended. Use a virtual environment.

```bash
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ml_experiments.py
python -m pytest -q
```


### Reproducible experiment reports

Running `python ml_experiments.py` generates:

- `reports/results.json` — Model comparison, final selected-model evaluation, complexity diagnostics, and subgroup results.
- `reports/tree_complexity.png` — Training versus cross-validation accuracy across decision-tree depths.

GitHub Actions automatically runs the test suite and experiments on every push.

To inspect the latest generated reports without running the project locally:

1. Open the repository's **Actions** tab.
2. Select the latest successful **ML Pipeline Tests** run.
3. Download the `ml-experiment-reports` artifact from the **Artifacts** section.

The committed report files may reflect an earlier run; workflow artifacts contain the results generated by their corresponding CI execution.


For the short student-facing activity, run:

```bash
python classification_demo.py
```

## Project structure

```text

ml-classification-workshop/
├── .github/
│   └── workflows/
│       └── ml-tests.yml
├── ml_experiments.py
├── classification_demo.py
├── tests/
│   └── test_ml_experiments.py
├── reports/
│   ├── results.json
│   └── tree_complexity.png
├── requirements.txt
└── README.md

```

## Methodological decisions & limitations

- **Data splitting:** Stratify by class to preserve class proportions. Reserve the test set for evaluation rather than hyperparameter decisions.
- **Preprocessing and leakage:** Scale features in `Pipeline`; the scaler is fitted per training fold in cross-validation. Trees require no feature scaling.
- **Model comparison:** Use the same CV partition and selection metric for every candidate. CV scores here are best-configuration scores from the search and can be optimistic; a held-out test helps assess selected-model performance.
- **Precision/recall:** Binary metrics take `dog` as the positive class; the 50/50 class split makes a majority-class accuracy baseline 50%, but F1 is zero when it predicts no dogs.
- **Overfitting experiment:** Inject known synthetic label noise only into the complexity study's training data to expose memorization. The benchmark uses clean labels. The studies have different purposes and should not be interpreted as the same training run.
- **Dataset limitations:** Two numeric features, simplified Gaussian distributions, and small subgroup test counts cannot establish real-world reliability, demographic fairness, or deployment readiness.
- **Next improvements:** Repeated CV / uncertainty estimates, a group-preserving ablation with controlled sample counts, learning curves, additional diagnostics, and better synthetic domain shifts.

## Teaching application

The companion classroom script reduces the concepts to three questions: *How does a model learn? When does it memorize? Who gets left out when training examples are missing?* It is intentionally simpler than the technical experiment. The unfinished team proposal is excluded from this repository.
