import numpy as np
import util
from sklearn.naive_bayes import GaussianNB

# Naive Bayes imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from util import dataprocessing as dp
from sklearn import metrics

# SVC imports
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Linear SVC imports
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# SGD imports
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main(save_path, train_path, pos):
    """
    High bias
    Validation Accuracy:                Training Accuracy:
        WR: 0.5329896907216495              WR: 0.5394575790762095      
        TE: 0.5181159420289855              TE: 0.5077027639329407      up
        RB: 0.5538881309686221              RB: 0.5527886747398942      up
    """
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(train_path, pos, add_intercept=True)


    # Naive Bayes
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    GaussianNB()



    # SVC:  WR: (v: 0.5363583478766725, t: 0.5479272727272727)      same
    #       TE: (v: 0.5848849945235487, t: 0.5653661875427789)      same
    #       RB: (v: 0.5614973262032086, t: 0.5830230115535185)      v up t down (basically swapped)
    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='poly'))
    # clf.fit(x_train, y_train)


    # Linear SVC:   WR: (v: 0.5136707388016288, t: 0.5296)
    #               TE: (v: 0.5662650602409639, t: 0.5620807665982204)
    #               RB: (v: 0.5630252100840336, 0.5722333619784207)
    '''clf = make_pipeline(StandardScaler(), LinearSVC(dual="auto", random_state=0, tol=1e-5))
    clf.fit(x_train, y_train)'''
    #print(clf.named_steps['linearsvc'].coef_)
    #print(clf.named_steps['linearsvc'].intercept_)


    # SGD Classifier:   WR: (v: 0.5119255381035486, t: 0.49527272727272725)
    #                   TE: (v: 0.5377875136911281, t: 0.5373032169746749)
    #                   RB: (v: 0.5637891520244461, t: 0.5692733696171106)
    '''clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(x_train, y_train)'''
    #print(clf.named_steps['sgdclassifier'].coef_)
    #print(clf.named_steps['sgdclassifier'].intercept_)

    # validation accuracy: (wr, ), (te, ), (rb, )
    results = clf.predict(x_train)
    print("Test Accuracy:", metrics.accuracy_score(y_train, results))
    print("F1 score:", metrics.f1_score(y_test, results))
    print("Recall Score:", metrics.recall_score(y_test, results))
    print("Precision score:", metrics.precision_score(y_test, results))


    # training accuracy: (wr, ), (te, ), (rb, )
    #train_results = clf.predict(x_train)
    #print("Training Accuracy:", metrics.accuracy_score(y_train, train_results))

    


if __name__ == '__main__':
    main()