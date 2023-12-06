import sys
import numpy as np

from util import dataprocessing as dp
from util import plots

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC

def main(save_path, train_path, pos):
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(train_path, pos, add_intercept=True)
    abc = AdaBC()
    abc.get_fit(x_train, y_train)
    y_pred = abc.get_accuracy(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # some sort of plotting function here
class AdaBC:
    def __init__(self, n_estimators=50, learning_rate=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = AdaBoostClassifier(n_estimators=n_estimators, base_estimator=SVC(probability=True, kernel='linear'), learning_rate=learning_rate)
    def get_fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    def get_accuracy(self, x_test):
        return self.model.predict(x_test)

if __name__ == '__main__':
    main()

