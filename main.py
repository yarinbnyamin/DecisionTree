from my_id3 import MyID3
from my_bagging_id3 import MyBaggingID3

from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_breast_cancer


# pre proccess datasets
def normalize(X):
    for column in X.columns:
        X[column] = X[column] / X[column].abs().max()
    return X.to_numpy()


data_base = {}

# add one dataset to the run
breast_cancer_ds = load_breast_cancer()
X = pd.DataFrame(breast_cancer_ds.data)
X = normalize(X)
X, y = (X > 0.5).astype(int), breast_cancer_ds.target
data_base["breast_cancer"] = (X, y)

# init algorithms
my_id3 = MyID3(max_depth=2)

my_bagging_id3 = MyBaggingID3(
    n_estimators=10, max_samples=0.5, max_features=0.2, max_depth=2
)

# compare algorithms
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
d = {
    "Dataset": [],
    "Method": [],
    "Evaluation metric": [],
    "Evaluation Value": [],
    "Fit Runtime (in ms)": [],
}

for name, (X, y) in data_base.items():
    for score in scoring:
        my_id3_score = []
        my_bagging_id3_score = []

        my_id3_time = []
        my_bagging_id3_time = []

        for i in range(3):
            t1 = time.time()
            my_id3_score.append(
                np.mean(cross_val_score(my_id3, X, y, cv=10, scoring=score))
            )
            t2 = time.time()
            my_id3_time.append(t2 - t1)

            t1 = time.time()
            my_bagging_id3_score.append(
                np.mean(cross_val_score(my_bagging_id3, X, y, cv=10, scoring=score))
            )
            t2 = time.time()
            my_bagging_id3_time.append(t2 - t1)

        my_id3_score = np.mean(my_id3_score)
        my_bagging_id3_score = np.mean(my_bagging_id3_score)

        # mean of 3 times of 10 k fold in sec * 1000 (convert to ms) / 10 (for one fit)
        my_id3_time = np.mean(my_id3_time) * 1000 / 10
        my_bagging_id3_time = np.mean(my_bagging_id3_time) * 1000 / 10

        d["Dataset"].append(name)
        d["Method"].append("my_id3")
        d["Evaluation metric"].append(score)
        d["Evaluation Value"].append(my_id3_score)
        d["Fit Runtime (in ms)"].append(my_id3_time)

        d["Dataset"].append(name)
        d["Method"].append("my_bagging_id3")
        d["Evaluation metric"].append(score)
        d["Evaluation Value"].append(my_bagging_id3_score)
        d["Fit Runtime (in ms)"].append(my_bagging_id3_time)


# export data
df = pd.DataFrame(data=d)
df.to_csv("summary.csv")
