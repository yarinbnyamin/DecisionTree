from my_id3 import MyID3
import numpy as np
from collections import Counter


class MyBaggingID3:
    def __init__(self, n_estimators, max_samples, max_features, max_depth):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes_ = sorted(set(y))
        self.estimators = []

        # Train each base estimator
        for _ in range(self.n_estimators):
            # Sample data with replacement
            sample_indices = np.random.choice(
                num_samples, int(self.max_samples * num_samples), replace=True
            )
            X_sampled, y_sampled = X[sample_indices], y[sample_indices]

            # Sample features without replacement
            feature_indices = np.random.choice(
                num_features, int(self.max_features * num_features), replace=False
            )
            X_sampled = X_sampled[:, feature_indices]

            # Train base estimator
            estimator = MyID3(self.max_depth)
            estimator.fit(X_sampled, y_sampled)
            self.estimators.append((estimator, feature_indices))

        # Return the classifier
        return self

    def predict(self, X):
        predictions = []
        for estimator, feature_indices in self.estimators:
            X_sampled = X[:, feature_indices]
            predictions.append(estimator.predict(X_sampled))

        # Aggregate predictions by majority vote
        return np.apply_along_axis(
            lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions
        )

    def predict_proba(self, X):
        probabilities = []
        for estimator, feature_indices in self.estimators:
            X_sampled = X[:, feature_indices]
            probabilities.append(estimator.predict_proba(X_sampled))

        # Average probabilities
        return np.mean(probabilities, axis=0)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "max_depth": self.max_depth,
        }
