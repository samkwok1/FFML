import sys
import numpy as np

from util import dataprocessing as dp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

def main(save_path, train_path, pos):
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(train_path, pos, add_intercept=True)
    y_valid = y_valid.squeeze()
    y_train = y_train.squeeze()
    # L1 regularization (lasso)
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.01)

    # L2 regularization (ridge)
    l2_model = LogisticRegression(penalty='l2', solver='liblinear', C=0.01)

    # Elastic-Net regularization
    elastic_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.01)

    print("L1 Model:")
    l1_model.fit(x_train, y_train)
    y_pred = l1_model.predict(x_valid)  

    conf_matrix = confusion_matrix(y_valid, y_pred)

    TN, FP, FN, TP = conf_matrix.ravel()

    # False Positive Rate = FP / (FP + TN)
    FPR = FP / (FP + TN)

    # False Negative Rate = FN / (FN + TP)
    FNR = FN / (FN + TP)
    mse = mean_squared_error(y_valid, y_pred)  
    print(f'Mean Squared Error: {mse}')
    print(f"False Positive Rate: {FPR}")
    print(f"False Negative Rate: {FNR}")
    print("Accuracy:",accuracy_score(y_valid, y_pred))

    print("**************\n")

    print("L2 Model:")
    l2_model.fit(x_train, y_train)
    y_pred = l2_model.predict(x_valid)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    # False Positive Rate = FP / (FP + TN)
    FPR = FP / (FP + TN)

    # False Negative Rate = FN / (FN + TP)
    FNR = FN / (FN + TP)

    mse = mean_squared_error(y_valid, y_pred)  
    print(f'Mean Squared Error: {mse}')
    print(f"False Positive Rate: {FPR}")
    print(f"False Negative Rate: {FNR}")
    print("Accuracy:",accuracy_score(y_valid, y_pred))
    print("**************\n")

    print("Elastic Model:")
    elastic_model.fit(x_train, y_train)
    y_pred = elastic_model.predict(x_train)
    # conf_matrix = confusion_matrix(y_valid, y_pred)
    # TN, FP, FN, TP = conf_matrix.ravel()

    # # False Positive Rate = FP / (FP + TN)
    # FPR = FP / (FP + TN)

    # # False Negative Rate = FN / (FN + TP)
    # FNR = FN / (FN + TP)

    # mse = mean_squared_error(y_valid, y_pred)  
    # print(f'Mean Squared Error: {mse}')
    # print(f"False Positive Rate: {FPR}")
    # print(f"False Negative Rate: {FNR}")
    print("Accuracy:",accuracy_score(y_train, y_pred))
    print("**************\n")


if __name__ == '__main__':
    main()
