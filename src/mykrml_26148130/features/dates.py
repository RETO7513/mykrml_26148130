import pandas as pd


def convert_to_date(df, cols):
    """Convert the specified columns of a dataframe into datetime

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : list
        List of column names to convert to datetime

    Returns
    -------
    pd.DataFrame
        Dataframe with the specified columns converted to datetime
    """
    df_copy = df.copy()
    for col in cols:
        df_copy[col] = pd.to_datetime(df_copy[col])

    return df_copy