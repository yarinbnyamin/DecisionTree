import math
import numpy as np
from collections import Counter


class MyID3:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Args:
        X (ndarray): Data matrix of shape(n_samples, n_features)
        y (array like): list or ndarray with n_samples containing the target variable
        """

        if isinstance(X[0][0], np.complex128) or isinstance(y[0], np.complex128):
            raise ValueError("Complex data not supported")

        self.classes_ = sorted(set(y))
        self.tree = self._build_tree(X, y)

        # Return the classifier
        return self

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(set(y))

        # Stop conditions
        if depth == self.max_depth or num_samples < 2 or num_classes == 1:
            return Counter(y)

        # Select the best feature to split the data
        node_indices = range(num_samples)
        best_feature = self.get_best_split(X, y, node_indices)

        # Split the data using the best feature and value
        left_idx = X[:, best_feature] == 1
        right_idx = X[:, best_feature] == 0
        left_subset, left_labels = X[left_idx], y[left_idx]
        right_subset, right_labels = X[right_idx], y[right_idx]
        tree = {"feature": best_feature, "children": {}}
        tree["children"][1] = self._build_tree(
            left_subset, left_labels, depth + 1
        )  # left
        tree["children"][0] = self._build_tree(
            right_subset, right_labels, depth + 1
        )  # right

        return tree

    def predict_proba(self, X):
        """
        Args:
        X (ndarray): Data matrix of shape(n_samples, n_features)

        Returns:
        Prob (ndarray) of shape (n_samples, n_classes)
        """
        proba = np.zeros((len(X), len(self.classes_)))
        lst = [self._traverse_tree(x, self.tree) for x in X]
        for i, x in enumerate(lst):
            for c in self.classes_:
                proba[i, c] = x[c] / sum(x.values())

        return proba

    def predict(self, X):
        """
        Args:
        X (ndarray): Data matrix of shape(n_samples, n_features)

        Returns:
        y (array like) of shape (n_samples,)
        """

        lst = [self._traverse_tree(x, self.tree) for x in X]
        lst = [x.most_common(1)[0][0] for x in lst]
        return np.array(lst)

    def _traverse_tree(self, x, tree):
        if isinstance(tree, Counter):
            return tree

        feature = tree["feature"]
        value = x[feature]
        subtree = tree["children"][value]

        if isinstance(subtree, Counter) and subtree == Counter():
            subtree = tree["children"][int(not value)]

        return self._traverse_tree(x, subtree)

    def get_params(self, deep=True):
        return {"max_depth": self.max_depth}

    @staticmethod
    def compute_entropy(y):
        """
        Computes the entropy for Args:
        y (ndarray): Numpy array indicating whether each example at a node is positive (`1`) or negative (`0`)
        Returns: entropy (float): Entropy at that node
        """

        if len(y) == 0:
            return 0

        # Compute the fraction of positive examples
        p1 = (y == 1).sum() / len(y)

        if p1 == 0 or p1 == 1:
            return 0

        # Calculate entropy (H)
        entropy = -p1 * math.log2(p1) - (1 - p1) * math.log2(1 - p1)

        return entropy

    @staticmethod
    def split_dataset(X, node_indices, feature):
        """
        Splits the data at the given node into
        left and right branches

        Args:
            X (ndarray): Data matrix of shape(n_samples, n_features)
            node_indices (list): List containing the active indices. I.e, the samples being considered at this step.
            feature (int): Index of feature to split on

        Returns:
            left_indices (list): Indices with feature value == 1
            right_indices (list): Indices with feature value == 0
        """

        # You need to return the following variables correctly
        left_indices = []
        right_indices = []

        for i in node_indices:
            if X[i, feature] == 1:
                left_indices.append(i)
            else:
                right_indices.append(i)

        return left_indices, right_indices

    @staticmethod
    def compute_information_gain(X, y, node_indices, feature):
        """
        Compute the information of splitting the node on a given feature

        Args:
            X (ndarray): Data matrix of shape(n_samples, n_features)
            y (array like): list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

        Returns:
            cost (float): Cost computed
        """

        # Compute entropy of the node_indices
        entropy_node = MyID3.compute_entropy(y[node_indices])

        # Compute entropy of the left and right branches
        left_indices, right_indices = MyID3.split_dataset(X, node_indices, feature)
        entropy_left = MyID3.compute_entropy(y[left_indices])
        entropy_right = MyID3.compute_entropy(y[right_indices])

        # Compute information gain
        information_gain = entropy_node - (
            (len(left_indices) / len(node_indices)) * entropy_left
            + (len(right_indices) / len(node_indices)) * entropy_right
        )

        return information_gain

    @staticmethod
    def get_best_split(X, y, node_indices):
        """
        Returns the optimal feature and threshold value
        to split the node data

        Args:
            X (ndarray): Data matrix of shape(n_samples, n_features)
            y (array like): list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.


        Returns:
            best_feature (int): The index of the best feature to split
        """

        # Initialize variables
        best_gain = -1
        best_feature = -1
        n_features = X.shape[1]

        # Loop over all features
        for feature in range(n_features):
            # Compute the information gain
            gain = MyID3.compute_information_gain(X, y, node_indices, feature)

            # Compare to current best gain
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        return best_feature
