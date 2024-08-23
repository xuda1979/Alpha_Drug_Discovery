from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k='all'):
    """
    Select and scale features for model training.

    Parameters:
    X (pd.DataFrame): Features data.
    y (pd.Series): Target variable.
    k (int or 'all'): Number of top features to select, based on univariate statistical tests.

    Returns:
    np.ndarray: Scaled and selected features.
    """
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    return X_scaled
