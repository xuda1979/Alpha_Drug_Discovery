from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def generate_features(X, y=None, k="all"):
    """Select features and apply scaling.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like, optional
        Target vector used for supervised feature selection.
    k : int or "all"
        Number of top features to keep.

    Returns
    -------
    ndarray
        Transformed feature array.
    """
    if y is not None:
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
    else:
        X_selected = X
    scaler = StandardScaler()
    return scaler.fit_transform(X_selected)
