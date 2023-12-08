import sys
import numpy as np

from util import dataprocessing as dp
from util import plots

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def main(save_path, train_path, pos):
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(train_path, pos, add_intercept=True)
    abc = AdaBC()
    abc.get_fit(x_train, np.squeeze(y_train))
    y_pred = abc.get_accuracy(x_train)

    print("Train Accuracy:", metrics.accuracy_score(y_train, y_pred))

    y_pred = abc.get_accuracy(x_test)
    print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("F1 score:", metrics.f1_score(y_test, y_pred))
    print("Recall Score:", metrics.recall_score(y_test, y_pred))
    print("Precision score:", metrics.precision_score(y_test, y_pred))
    # some sort of plotting function here
class AdaBC:
    def __init__(self, n_estimators=50, learning_rate=0.5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # self.model = AdaBoostClassifier(n_estimators=n_estimators, base_estimator=DecisionTreeClassifier(max_depth=2), learning_rate=learning_rate)
        self.model = RandomForestClassifier(n_estimators=100, max_depth = 5, random_state=42)
    def get_fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    def get_accuracy(self, x_test):
        return self.model.predict(x_test)

if __name__ == '__main__':
    main()

