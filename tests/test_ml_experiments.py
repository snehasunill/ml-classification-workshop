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

    candidates = [
        'dummy_majority',
        'logistic_regression',
        'knn',
        'decision_tree',
    ]

    # Every candidate must have CV results.
    for name in candidates:
        assert name in result
        assert 0 <= result[name]['cv_f1_mean'] <= 1
        assert 'best_params' in result[name]

        # Unselected candidates should not have test metrics.
        assert 'test' not in result[name]
        assert 'test_metrics' not in result[name]

    # Selection must correspond to the highest CV F1.
    selected = result['_selection']['selected_using_training_cv']
    assert selected in candidates

    best_cv_f1 = max(result[name]['cv_f1_mean'] for name in candidates)
    assert result[selected]['cv_f1_mean'] == best_cv_f1

    # Final test metrics belong only to the selected model.
    test_metrics = result['_selection']['test_metrics']
    assert all(0 <= value <= 1 for value in test_metrics.values())

    # Confusion matrix must account for every test observation.
    cm = result['_selection']['test_confusion_matrix']
    assert len(cm) == 2
    assert all(len(row) == 2 for row in cm)
    assert sum(map(sum, cm)) == len(test)

    assert result['_selection']['test_confusion_matrix_labels'] == [
        'cat', 'dog'
    ]



def test_complexity_and_representational_slices():
    train, test = split_data(make_dataset())
    curves = complexity_study(train)
    assert len(curves) == 6
    assert all(0 <= row['cv_accuracy'] <= 1 for row in curves)
    slices = representation_study(train, test)
    assert len(slices) == 6
    assert all(row['n_test'] > 0 for row in slices)
