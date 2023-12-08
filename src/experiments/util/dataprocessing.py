import numpy as np
import pandas as pd
from math import isnan
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

import sys
def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x
    return new_x

def load_dataset(filename, position, add_intercept):
    """
    Takes in a dataset and deletes the unnecessary columns. This includes all columns that 
    contain non-projection information, half and zero PPR columns, and columns that have the
    difference of two other columns. The new array will just have features:
    ProjRushYd/ProjRushAtt, ProjRushTD, ProjRecYd/ProjRecCount, ProjRecTD, DiffPPR1

    @param: filename of dataset (csv)
    @param: position (wr, te, rb)
    @return: x (n_features, n_examples)
    @return: y (n_examples, )
    """
    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)


    scaler = PowerTransformer(method='yeo-johnson')  # or method='box-cox'
    standard_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    # Enumerate the features we want
    allowed_col_labels = ['ProjRushAtt','ProjRushYd','RushingYardsPerAttempt','ProjRushTD','ReceivingTargets','ProjRecCount','ProjRecYd','ProjRecTD','Fumbles','FumblesLost','DiffPPR1']

    # Load Dataset
    df = pd.read_csv(filename)

    # Select the features we want
    new_df = df[allowed_col_labels].copy()

    # Change last column to labels
    new_df['DiffPPR1'] = new_df['DiffPPR1'].apply(lambda x: 0 if x < 0 else 1)

    new_df_0 = new_df[new_df["DiffPPR1"] == 0]
    new_df_1 = new_df[new_df["DiffPPR1"] == 1]

    df_class_0_bal = resample(new_df_0, replace=False, n_samples=len(new_df_1), random_state=42)

    new_df = pd.concat([df_class_0_bal, new_df_1])

    new_df_0 = new_df[new_df["DiffPPR1"] == 0]
    new_df_1 = new_df[new_df["DiffPPR1"] == 1]

    # # Log Scaling
    # log_scaled = np.log1p(new_df.iloc[:,:-1])

    # Yeo-Johnson transformation
    yeo_johnson_transformed = scaler.fit_transform(new_df.iloc[:,:-1])

    # Standard Scaler
    standard_scaled = standard_scaler.fit_transform(yeo_johnson_transformed)

    # Min-Max Scaler
    min_max_scaled = min_max_scaler.fit_transform(standard_scaled)

    # Recombine with the labels
    new_df.iloc[:,:-1] = yeo_johnson_transformed
    
    # Splitting 80% for training and 20% for temp (which will be further split)
    train_df, temp_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df["DiffPPR1"])

    # Splitting the temp_df into 50% validation and 50% test (which results in 10% of original data for both)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["DiffPPR1"])


    x_train, y_train = train_df.iloc[:,:-1].values, train_df.iloc[:, [-1]].values
    x_valid, y_valid = valid_df.iloc[:,:-1].values, valid_df.iloc[:, [-1]].values
    x_test, y_test = test_df.iloc[:,:-1].values, test_df.iloc[:, [-1]].values
    # Isolate and return features and labels (x_train, y_train, x_valid, y_valid, x_test, y_test)

    if add_intercept:
        x_train = add_intercept_fn(np.array(x_train))
        x_valid = add_intercept_fn(np.array(x_valid))
        x_test = add_intercept_fn(np.array(x_test))

    return x_train, np.array(y_train), x_valid, np.array(y_valid), x_test, np.array(y_test)


# made a main function so I could debug this easily

def main():
    args = sys.argv[1:]
    load_dataset(args[0], args[1])

if __name__ == "__main__":
    main()
