import pandas as pd
import sys
import csv

def make_train_data_pandas(pos):
    if pos == 'rb':
        proj = "../input_data/RBs/rb_13-22_projections_sorted.csv"
        actual = "../input_data/RBs/rb_13-22_real_sorted.csv"
        output = "../input_data/RBs/rb_13-22_final.csv"
    if pos == 'wr':
        proj = "../input_data/WRs/wr_13-22_projections_sorted.csv"
        actual = "../input_data/WRs/wr_13-22_real_sorted.csv"
        output = "../input_data/WRs/wr_13-22_final.csv"
    if pos == 'te':
        proj = "../input_data/TEs/te_13-22_projections_sorted.csv"
        actual = "../input_data/TEs/te_13-22_real_sorted.csv"
        output = "../input_data/Tes/te_13-22_final.csv"

    df_proj = pd.read_csv(proj, header=0)
    df_actual = pd.read_csv(actual, header=0)
    shared_chars = ['Name', 'Team', 'Position', 'Week', 'Opponent']
    df_mask = (df_proj['RushingAttempts'] == 0) & (df_proj['RushingYards'] == 0) & (df_proj['RushingYardsPerAttempt'] == 0) & (df_proj['RushingTouchdowns'] == 0) & (df_proj['ReceivingTargets'] == 0) & (df_proj['Receptions'] == 0) & (df_proj['ReceivingYards'] == 0) & (df_proj['ReceivingTouchdowns'] == 0)
    df_proj = df_proj[~df_mask]
    trimmed_df_proj = df_proj.merge(df_actual[shared_chars], on=shared_chars, how='inner')
    trimmed_df_actual = df_actual.merge(df_proj[shared_chars], on=shared_chars, how='inner')
    labels_vector = trimmed_df_actual.iloc[:, -1] - trimmed_df_proj.iloc[:, -1]
    trimmed_df_proj.iloc[:, -1] = labels_vector
    ['ProjRushYd', 'ProjRushAtt', 'ProjRushTD', 'ProjRecYd', 'ProjRecCount', 'ProjRecTD', 'DiffPPR1']
    trimmed_df_proj = trimmed_df_proj.rename(columns={"FantasyPoints": "DiffPPR1",
                                                    "RushingAttempts": 'ProjRushAtt',
                                                    "RushingYards": 'ProjRushYd',
                                                    "RushingTouchdowns": 'ProjRushTD',
                                                    "Receptions": 'ProjRecCount',
                                                    "ReceivingTouchdowns": "ProjRecTD",
                                                    "ReceivingYards": "ProjRecYd"})

    

    trimmed_df_proj.to_csv(output, index=False)

def make_train_data_manual(pos):
    if pos == 'rb':
        proj = "../input_data/RBs/rb_13-22_projections_sorted.csv"
        actual = "../input_data/RBs/rb_13-22_real_sorted.csv"
    if pos == 'wr':
        proj = "../input_data/WRs/wr_13-22_projections_sorted.csv"
        actual = "../input_data/WRs/wr_13-22_real_sorted.csv"
    if pos == 'te':
        proj = "../input_data/TEs/te_13-22_projections_sorted.csv"
        actual = "../input_data/TEs/te_13-22_real_sorted.csv"
    df_proj = pd.read_csv(proj)
    df_actual = pd.read_csv(actual)

    mask = []
    index = 0
    for proj_ind, row in df_proj.iterrows():
        if (df_actual.iloc[index].loc['Name'] == row.loc['Name']) and (df_actual.iloc[index].loc['Team'] == row.loc['Team']) and (df_actual.iloc[index].loc['Week'] == row.loc['Week']) and (df_actual.iloc[index].loc['Opponent'] == row.loc['Opponent']):
            mask.append(True)
            index += 1
        else:
            mask.append(False)
            temp_ind = proj_ind
            while not ((df_actual.iloc[index].loc['Name'] == df_proj.iloc[temp_ind].loc['Name']) and (df_actual.iloc[index].loc['Team'] == df_proj.iloc[temp_ind].loc['Team']) and (df_actual.iloc[index].loc['Week'] == df_proj.iloc[temp_ind].loc['Week']) and (df_actual.iloc[index].loc['Opponent'] == df_proj.iloc[temp_ind].loc['Opponent'])):
                if (df_actual.iloc[index].loc['Name'] != df_proj.iloc[temp_ind].loc['Name']):
                    index += 1
                    break 
                temp_ind += 1
    trimmed_df = df_proj[mask]

def main():
    args = sys.argv[1:]
    make_train_data_pandas(args[0])
    # make_train_data_manual(args[0])

if __name__ == '__main__':
    main()