import numpy as np
import util
from sklearn.naive_bayes import GaussianNB

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from util import dataprocessing as dp
from sklearn import metrics

rb_train_path = "src/input_data/RBs/all_rb_stats.csv"
te_train_path = "src/input_data/tight_ends/all_te_stats.csv"
wr_train_path = "src/input_data/wrs/all_wr_stats.csv"

def main(save_path, train_path, pos):
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(train_path, pos, add_intercept=True)
    
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    GaussianNB()
    results = clf.predict(x_valid)
    print("Accuracy:", np.mean(results == y_valid))

    print((sum(results), np.sum(y_valid)))

if __name__ == '__main__':
    main()