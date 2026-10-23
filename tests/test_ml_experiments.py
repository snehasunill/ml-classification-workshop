from ml_experiments import make_dataset, split_data, benchmark, complexity_study, representation_study


def test_reproducible_generation():
    assert make_dataset(seed=4).equals(make_dataset(seed=4))
    assert not make_dataset(seed=4).equals(make_dataset(seed=5))


def test_disjoint_holdout_and_stratification():
    train, test = split_data(make_dataset())
    assert set(train.index).isdisjoint(test.index)
    assert train['pet'].value_counts().to_dict() == {'cat': 360, 'dog': 360}
    assert test['pet'].value_counts().to_dict() == {'cat': 90, 'dog': 90}


def test_cv_model_selection_and_metrics():
    train, test = split_data(make_dataset(n_per_class=100))
    result = benchmark(train, test)
    assert result['_selection']['selected_using_training_cv'] in ['dummy_majority', 'logistic_regression', 'knn', 'decision_tree']
    assert sum(map(sum, result['_selection']['test_confusion_matrix'])) == len(test)
    assert all(0 <= row['test']['f1_dog'] <= 1 for name, row in result.items() if not name.startswith('_'))


def test_complexity_and_representational_slices():
    train, test = split_data(make_dataset())
    curves = complexity_study(train)
    assert len(curves) == 6
    assert all(0 <= row['cv_accuracy'] <= 1 for row in curves)
    slices = representation_study(train, test)
    assert len(slices) == 6
    assert all(row['n_test'] > 0 for row in slices)
