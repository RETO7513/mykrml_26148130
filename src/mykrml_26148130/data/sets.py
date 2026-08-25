def pop_target(df, target_col):
    """Extract target variable from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column

    Returns
    -------
    pd.DataFrame
        Features (input dataframe with target removed)
    pd.Series
        Target variable
    """
    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target