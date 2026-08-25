from sklearn.metrics import mean_squared_error, mean_absolute_error


def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided predictions and actual values

    Parameters
    ----------
    y_preds : np.ndarray or pd.Series
        Predicted target values
    y_actuals : np.ndarray or pd.Series
        Actual target values
    set_name : str, optional
        Name of the set (e.g. 'Training', 'Validation') for labelling the output

    Returns
    -------
    None
    """
    rmse = mean_squared_error(y_actuals, y_preds, squared=False)
    mae = mean_absolute_error(y_actuals, y_preds)

    print(f"RMSE {set_name}: {rmse}")
    print(f"MAE {set_name}: {mae}")