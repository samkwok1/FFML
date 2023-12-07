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
    nn = NeuralNetwork()
    nn.get_fit(x_train, y_train)
    y_pred = nn.get_accuracy(x_valid)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

class NeuralNetwork:
    def __init__(self, hidden_sizes=[5], regularization=0.0001, batch_size=200):
        self.hidden_sizes = hidden_sizes
        self.alpha = regularization
        self.batch_size = batch_size
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_sizes, solver='sgd', alpha=self.alpha, batch_size=self.batch_size, learning_rate='adaptive')

    def get_fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def get_accuracy(self, x_eval):
        return self.model.predict(x_eval)



if __name__ == '__main__':
    main()