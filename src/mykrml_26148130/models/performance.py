def print_classifier_scores(y_preds, y_actuals, set_name=None):
    """Print Accuracy and F1 for the provided data.

    F1 'average' is chosen automatically: 'binary' for binary targets,
    'weighted' for multi-class.

    Parameters
    ----------
    y_preds : array-like
        Predicted target (class labels)
    y_actuals : array-like
        Actual target
    set_name : str
        Name of the set to be printed
    """
    from sklearn.metrics import accuracy_score, f1_score
    import pandas as pd

    average = 'weighted' if pd.Series(y_actuals).nunique() > 2 else 'binary'
    print(f"Accuracy {set_name}: {accuracy_score(y_actuals, y_preds)}")
    print(f"F1 {set_name}: {f1_score(y_actuals, y_preds, average=average)}")


def print_auprc(y_scores, y_actuals, set_name=None):
    """Print the AUPRC (Average Precision) for probability/score predictions.

    Use this with predicted probabilities or decision scores, NOT hard labels,
    because AUPRC is threshold-independent and needs a ranking.

    Parameters
    ----------
    y_scores : array-like
        Predicted probabilities or scores for the positive class
    y_actuals : array-like
        Actual target (0/1)
    set_name : str
        Name of the set to be printed
    """
    from sklearn.metrics import average_precision_score
    print(f"AUPRC {set_name}: {average_precision_score(y_actuals, y_scores)}")


def assess_classifier_set(model, features, target, set_name=''):
    """Predict with a trained model and print Accuracy and F1 on the given set.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    features : array-like
        Features
    target : array-like
        Target variable
    set_name : str
        Name of the set to be printed
    """
    preds = model.predict(features)
    print_classifier_scores(y_preds=preds, y_actuals=target, set_name=set_name)


def fit_assess_classifier(model, X_train, y_train, X_val, y_val):
    """Train a classifier, print Accuracy and F1 on train and validation, return the model.

    Parameters
    ----------
    model : sklearn estimator
        Instantiated model with set hyperparameters
    X_train, y_train : array-like
        Training features and target
    X_val, y_val : array-like
        Validation features and target

    Returns
    -------
    sklearn estimator
        The trained model
    """
    model.fit(X_train, y_train)
    assess_classifier_set(model, X_train, y_train, set_name='Training')
    assess_classifier_set(model, X_val, y_val, set_name='Validation')
    return model