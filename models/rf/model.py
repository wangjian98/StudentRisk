"""Random Forest model definition (sklearn)."""
from sklearn.ensemble import RandomForestClassifier


class RFModel:
    """Random Forest wrapper for binary classification.

    Uses sklearn's RandomForestClassifier. Default tuned for n=473 / 46-dim features.

    Convention: y=1 = FAILED (positive class).
    """
    def __init__(self, n_estimators: int = 200, max_depth: int = 10,
                 min_samples_split: int = 5, class_weight: str = 'balanced',
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None

    def fit(self, X_tr, y_tr, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_tr, y_tr)
        return self

    def predict_proba(self, X):
        """Return P(failed=1) as a 1D array of shape (n,)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)