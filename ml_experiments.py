"""Reproducible portfolio experiments on synthetic tabular classification data.

Run: python ml_experiments.py
The synthetic features are intentionally simple, not a proxy for real pets.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
FEATURES = ['weight_kg', 'ear_length_cm']
REPORTS = Path(__file__).parent / 'reports'


def make_dataset(seed: int = SEED, n_per_class: int = 450) -> pd.DataFrame:
    """Build reproducible overlapping classes, including three dog-size subgroups."""
    rng = np.random.default_rng(seed)
    cats = pd.DataFrame({
        'weight_kg': rng.normal(4.5, 1.0, n_per_class).clip(2, 8),
        'ear_length_cm': rng.normal(6, 1.0, n_per_class).clip(3, 9),
        'pet': 'cat', 'size': 'not_applicable',
    })
    sizes = rng.choice(['small', 'medium', 'large'], n_per_class, p=[0.35, 0.4, 0.25])
    weights = np.select([sizes == 'small', sizes == 'medium'],
                        [rng.normal(7, 1.5, n_per_class), rng.normal(18, 4, n_per_class)],
                        default=rng.normal(35, 6, n_per_class))
    ears = np.select([sizes == 'small', sizes == 'medium'],
                     [rng.normal(7, 1.2, n_per_class), rng.normal(9, 1.5, n_per_class)],
                     default=rng.normal(11, 1.5, n_per_class))
    dogs = pd.DataFrame({'weight_kg': weights.clip(3, 60),
                         'ear_length_cm': ears.clip(4, 16), 'pet': 'dog', 'size': sizes})
    return pd.concat([cats, dogs], ignore_index=True)


def split_data(df: pd.DataFrame):
    """Hold out 20% as test; CV and model selection use only the other 80%."""
    train, test = train_test_split(df, test_size=0.20, stratify=df['pet'], random_state=SEED)
    return train, test


def metrics(y_true, y_pred) -> dict:
    return {key: round(float(value), 4) for key, value in {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_dog': precision_score(y_true, y_pred, pos_label='dog', zero_division=0),
        'recall_dog': recall_score(y_true, y_pred, pos_label='dog'),
        'f1_dog': f1_score(y_true, y_pred, pos_label='dog'),
    }.items()}


def benchmark(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Tune using stratified CV within training only; evaluate once on held-out test."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        'dummy_majority': (DummyClassifier(strategy='most_frequent'), {}),
        'logistic_regression': (Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=SEED))]), {'clf__C': [0.01, 0.1, 1, 10]}),
        'knn': (Pipeline([('scale', StandardScaler()), ('clf', KNeighborsClassifier())]), {'clf__n_neighbors': [3, 5, 11, 21]}),
        'decision_tree': (DecisionTreeClassifier(random_state=SEED), {'max_depth': [2, 3, 5, 10, None], 'min_samples_leaf': [1, 5]}),
    }
    X_train, y_train = train[FEATURES], train['pet']
    X_test, y_test = test[FEATURES], test['pet']
    
    summary, fitted, cv_scores = {}, {}, {}
    

    # Step 1: Tune and compare models using training data only.
    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(
            estimator,
            grid or {},
            cv=cv,
            scoring=make_scorer(
                f1_score, pos_label='dog', zero_division=0
            ),
            refit=True,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        cv_scores[name] = float(search.best_score_)

        summary[name] = {
            'cv_f1_mean': round(float(search.best_score_), 4),
            'best_params': search.best_params_,
        }
        fitted[name] = search.best_estimator_

    # Step 2: Select the winner using cross-validation F1.
    best_name = max(cv_scores, key=cv_scores.get)

    # Step 3: Evaluate only the selected model on held-out data.
    best_pred = fitted[best_name].predict(X_test)

    summary['_selection'] = {
        'selected_using_training_cv': best_name,
        'test_metrics': metrics(y_test, best_pred),
        'test_confusion_matrix_labels': ['cat', 'dog'],
        'test_confusion_matrix': confusion_matrix(
            y_test, best_pred, labels=['cat', 'dog']
        ).tolist(),
    }

    return summary



def complexity_study(train: pd.DataFrame) -> list[dict]:
    """Train-vs-CV diagnostic; does not peek at the final test partition."""
    X, y = train[FEATURES], train['pet'].copy()
    # Flip training labels only to expose variance/overfitting of deep trees.
    rng = np.random.default_rng(SEED + 3)
    flip_pos = rng.choice(len(y), size=int(len(y) * .12), replace=False)
    y.iloc[flip_pos] = y.iloc[flip_pos].map({'cat': 'dog', 'dog': 'cat'})
    from sklearn.model_selection import cross_validate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for depth in [1, 2, 3, 5, 10, None]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        scores = cross_validate(model, X, y, cv=cv, scoring='accuracy', return_train_score=True)
        rows.append({'max_depth': 'unlimited' if depth is None else depth,
                     'train_accuracy': round(float(scores['train_score'].mean()), 4),
                     'cv_accuracy': round(float(scores['test_score'].mean()), 4)})
    return rows


def representation_study(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Controlled ablation: exclude small dogs vs. use full training set.

    Same model and held-out groups, but training sizes/class ratios differ;
    results show association, not an isolated causal effect of subgroup coverage.
    """
    limited = train[(train['pet'] == 'cat') | (train['size'] != 'small')]
    rows = []
    for label, data in [('small_dogs_excluded', limited), ('full_training_data', train)]:
        model = Pipeline([('scale', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))])
        model.fit(data[FEATURES], data['pet'])
        for size in ['small', 'medium', 'large']:
            group = test[(test['pet'] == 'dog') & (test['size'] == size)]
            rows.append({'training_condition': label, 'dog_size': size, 'n_test': len(group),
                         'recall_dog': round(float(recall_score(group['pet'], model.predict(group[FEATURES]), pos_label='dog', labels=['cat','dog'], zero_division=0)), 4)})
    return rows


def save_chart(rows: list[dict]) -> None:
    labels = [str(r['max_depth']) for r in rows]
    plt.figure(figsize=(7.6, 4.5))
    plt.plot(labels, [r['train_accuracy'] for r in rows], marker='o', label='Training accuracy (CV folds)')
    plt.plot(labels, [r['cv_accuracy'] for r in rows], marker='o', label='Validation accuracy (CV folds)')
    plt.xlabel('Decision tree max_depth')
    plt.ylabel('Mean accuracy')
    plt.title('Model complexity and generalization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / 'tree_complexity.png', dpi=150)
    plt.close()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    train, test = split_data(make_dataset())
    result = {'dataset': {'synthetic': True, 'n_train': len(train), 'n_test': len(test),
                          'features': FEATURES, 'seed': SEED},
              'model_comparison': benchmark(train, test),
              'complexity': complexity_study(train),
              'representation': representation_study(train, test)}
    (REPORTS / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    save_chart(result['complexity'])
    print('Model comparison (selection uses training-only 5-fold CV):')
    
    for name, row in result['model_comparison'].items():
        if not name.startswith('_'):
            print(f"{name:20s} CV F1={row['cv_f1_mean']:.3f}")

    selection = result['model_comparison']['_selection']
    print('Selected:', selection['selected_using_training_cv'])
    print('Selected model test metrics:', selection['test_metrics'])



if __name__ == '__main__':
    main()
