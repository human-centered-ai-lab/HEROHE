from sklearn.linear_model import LogisticRegression
from sklearn import set_config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


with open("/home/simon/PycharmProjects/robert_sql/slide_data_neg1_v2.txt") as fp:
    lines = fp.readlines()
    X_neg = np.zeros((len(lines)-1, len(lines[0].split(" "))))
    params = []
    for i, line in enumerate(lines):
        line_arr = line.replace("\n,", "").split(" ")
        for k, item in enumerate(line_arr):
            if k == 0:
                continue
            elif i == 0:
                params.append(item)
            elif k > 0:
                X_neg[i - 1, k- 1] = float(item)

y_neg = np.zeros(X_neg.shape[0])
lines = None
with open("/home/simon/PycharmProjects/robert_sql/slide_data_pos1_v2.txt") as fp:
    lines = fp.readlines()
    print(len(lines))
    X_pos = np.zeros((len(lines) - 1, len(lines[0].split(" "))))
    y_pos = np.zeros(len(lines))
    for i, line in enumerate(lines):
        line_arr = line.replace("\n,", "").split(" ")
        for k, item in enumerate(line_arr):
            if k == 0:
                continue
            elif i == 0:
                params.append(item)
            elif k > 0:
                X_pos[i - 1, k - 1] = float(item)

y_pos = np.ones(X_pos.shape[0])
X = np.concatenate((X_neg, X_pos))
print(X.shape)
y = np.concatenate((y_neg, y_pos))
print(X.shape, y.shape)

X, y = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

sol

lr = LogisticRegression(C=0.3, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=0.6, max_iter=5000,
                   multi_class='ovr', n_jobs=6, penalty="elasticnet",
                   random_state=None, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)

lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print(score)
