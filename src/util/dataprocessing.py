import sys
import numpy as np
import pandas as pd

"""
Takes in a dataset and deletes the unnecessary columns. This includes all columns that 
contain non-projection information, half and zero PPR columns, and columns that have the
difference of two other columns. The new array will just have features:
ProjRushYd/ProjRushAtt, ProjRushTD, ProjRecYd/ProjRecCount, ProjRecTD, DiffPPR1

@param: dataset
@return: np array
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


    print(new_df)


# made a main function so I could debug this easily

def main():
    args = sys.argv[1:]
    load_dataset('src/input_data/2023_rbs_wk1_thru_6/all_rb_stats.csv')

if __name__ == "__main__":
    main()
