from joblib import dump, load
import os

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

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally as .joblib files if they exist

    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame, optional
        Features for the training set
    y_train : np.ndarray or pd.Series, optional
        Target for the training set
    X_val : np.ndarray or pd.DataFrame, optional
        Features for the validation set
    y_val : np.ndarray or pd.Series, optional
        Target for the validation set
    X_test : np.ndarray or pd.DataFrame, optional
        Features for the testing set
    y_test : np.ndarray or pd.Series, optional
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    None
    """
    if X_train is not None:
        dump(X_train, f'{path}X_train.joblib')
    if X_val is not None:
        dump(X_val, f'{path}X_val.joblib')
    if X_test is not None:
        dump(X_test, f'{path}X_test.joblib')
    if y_train is not None:
        dump(y_train, f'{path}y_train.joblib')
    if y_val is not None:
        dump(y_val, f'{path}y_val.joblib')
    if y_test is not None:
        dump(y_test, f'{path}y_test.joblib')


def load_sets(path='../data/processed/'):
    """Load the different locally saved sets from the given folder if they exist

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Tuple containing (X_train, y_train, X_val, y_val, X_test, y_test)
    Any set that does not exist on disk is returned as None
    """
    X_train = load(f'{path}X_train.joblib') if os.path.isfile(f'{path}X_train.joblib') else None
    X_val   = load(f'{path}X_val.joblib')   if os.path.isfile(f'{path}X_val.joblib')   else None
    X_test  = load(f'{path}X_test.joblib')  if os.path.isfile(f'{path}X_test.joblib')  else None
    y_train = load(f'{path}y_train.joblib') if os.path.isfile(f'{path}y_train.joblib') else None
    y_val   = load(f'{path}y_val.joblib')   if os.path.isfile(f'{path}y_val.joblib')   else None
    y_test  = load(f'{path}y_test.joblib')  if os.path.isfile(f'{path}y_test.joblib')  else None

    return X_train, y_train, X_val, y_val, X_test, y_test

def subset_x_y(target, features, start_index, end_index):
    """Subset the target and features between the specified indexes

    Parameters
    ----------
    target : pd.Series or np.ndarray
        Target variable
    features : pd.DataFrame or np.ndarray
        Features
    start_index : int
        Index of the first row to include
    end_index : int
        Index of the last row to include

    Returns
    -------
    Subsetted features and target
    """
    return features[start_index:end_index], target[start_index:end_index]

def split_sets_by_time(features, target, test_ratio=0.2):
    """Split features and target into training, validation and testing sets by time order

    The data is assumed to already be sorted chronologically. The most recent rows
    become the testing set, the block before it the validation set, and the earliest
    rows the training set. Validation and testing sets have the same number of rows.

    Parameters
    ----------
    features : pd.DataFrame or np.ndarray
        Features (assumed sorted by time)
    target : pd.Series or np.ndarray
        Target variable (assumed sorted by time)
    test_ratio : float
        Proportion of rows to use for the testing set (default: 0.2)

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    num_total = len(target)
    num_test = int(num_total * test_ratio)
    num_val = int(num_total * test_ratio)
    num_train = num_total - num_val - num_test

    X_train, y_train = subset_x_y(target=target, features=features, start_index=0, end_index=num_train)
    X_val, y_val     = subset_x_y(target=target, features=features, start_index=num_train, end_index=num_train + num_val)
    X_test, y_test   = subset_x_y(target=target, features=features, start_index=num_train + num_val, end_index=num_total)

    return X_train, y_train, X_val, y_val, X_test, y_test
def split_sets_random(features, target, test_ratio=0.2, random_state=42):
    """Split features and target into training, validation and testing sets randomly

    The validation set is sized to have the same number of rows as the testing set.

    Parameters
    ----------
    features : pd.DataFrame or np.ndarray
        Features
    target : pd.Series or np.ndarray
        Target variable
    test_ratio : float
        Proportion of rows to use for the testing set (default: 0.2)
    random_state : int
        Seed for reproducible splits (default: 42)

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    from sklearn.model_selection import train_test_split

    val_ratio = test_ratio / (1 - test_ratio)

    X_data, X_test, y_data, y_test = train_test_split(
        features, target, test_size=test_ratio, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=val_ratio, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test