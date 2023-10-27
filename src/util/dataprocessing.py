import numpy as np
import pandas as pd
from math import isnan
from sklearn.model_selection import train_test_split

import sys

"""
Takes in a dataset and deletes the unnecessary columns. This includes all columns that 
contain non-projection information, half and zero PPR columns, and columns that have the
difference of two other columns. The new array will just have features:
ProjRushYd/ProjRushAtt, ProjRushTD, ProjRecYd/ProjRecCount, ProjRecTD, DiffPPR1

@param: dataset
@return: x (n_features, n_examples)
@return: y (n_examples, )
"""
def load_dataset(filename):
    # Enumerate the features we want
    allowed_col_labels = ['ProjRushYd', 'ProjRushAtt', 'ProjRushTD', 'ProjRecYd', 'ProjRecCount', 'ProjRecTD', 'DiffPPR1']

    # Load Dataset
    df = pd.read_csv(filename)

    # Select the features we want
    new_df = df[allowed_col_labels].copy()

    # Divide the needed columns
    new_df['ProjRushYd'] /= new_df['ProjRushAtt']
    new_df = new_df.drop('ProjRushAtt', axis=1)
    new_df.rename(columns={'ProjRushYd':'ProjRushYdPerAtt'}, inplace=True)

    new_df['ProjRecYd'] /= new_df['ProjRecCount']
    new_df = new_df.drop('ProjRecCount', axis=1)
    new_df.rename(columns={'ProjRecYd': 'ProjRecYdPerRec'}, inplace=True)

    # Change last column to labels
    new_df['DiffPPR1'] = new_df['DiffPPR1'].apply(lambda x: 0 if x < 0 else 1)

    # Handle NaN values from dividing by zero
    new_df['ProjRecYdPerRec'] = new_df['ProjRecYdPerRec'].apply(lambda x: 0 if isnan(x) else x)
    new_df['ProjRushYdPerAtt'] = new_df['ProjRushYdPerAtt'].apply(lambda x: 0 if isnan(x) else x)

    # Splitting 60% for training and 40% for temp (which will be further split)
    train_df, temp_df = train_test_split(new_df, test_size=0.4, random_state=42)

    # Splitting the temp_df into 50% validation and 50% test (which results in 20% of original data for both)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Isolate and return features and labels (x_train, y_train, x_valid, y_valid, x_test, y_test)
    x_train, y_train = train_df.iloc[:, [0, 1, 2, 3]].values, train_df.iloc[:, [4]].values
    x_valid, y_valid = valid_df.iloc[:, [0, 1, 2, 3]].values, valid_df.iloc[:, [4]].values
    x_test, y_test = test_df.iloc[:, [0, 1, 2, 3]].values, test_df.iloc[:, [4]].values
    return np.array(x_train), np.array(y_train), np.array(x_valid), np.array(y_valid), np.array(x_test), np.array(y_test)


# made a main function so I could debug this easily

def main():
    args = sys.argv[1:]
    load_dataset(args[0])

if __name__ == "__main__":
    main()
