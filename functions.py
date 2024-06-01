import math


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
    entropy_node = compute_entropy(y[node_indices])

    # Compute entropy of the left and right branches
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    entropy_left = compute_entropy(y[left_indices])
    entropy_right = compute_entropy(y[right_indices])

    # Compute information gain
    information_gain = entropy_node - (
        (len(left_indices) / len(node_indices)) * entropy_left
        + (len(right_indices) / len(node_indices)) * entropy_right
    )

    return information_gain


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
        gain = compute_information_gain(X, y, node_indices, feature)

        # Compare to current best gain
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature
