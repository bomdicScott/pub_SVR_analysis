from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
###############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

###############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

print("train data shape: %r, train target shape: %r"
      % (X_train.shape, y_train.shape)) # train data shape: (1437, 64), train target shape: (1437,)
print("test data shape: %r, test target shape: %r"
      % (X_test.shape, y_test.shape)) # test data shape: (360, 64), test target shape: (360,)
