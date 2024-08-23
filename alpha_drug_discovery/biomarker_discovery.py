# biomarker_discovery.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def discover_biomarkers(X, y, n_top_features=10):
    """
    Discover biomarkers using Random Forest feature importance.
    
    Parameters:
    X (pd.DataFrame): Omics data.
    y (pd.Series): Labels (e.g., disease vs. control).
    n_top_features (int): Number of top features to select as biomarkers.
    
    Returns:
    list: List of selected biomarkers.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    feature_importances = clf.feature_importances_
    top_features = X.columns[np.argsort(feature_importances)[-n_top_features:]]
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Biomarker discovery model accuracy: {accuracy * 100:.2f}%")
    
    return top_features.tolist()
