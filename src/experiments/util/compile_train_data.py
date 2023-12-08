import pandas as pd
import sys
import csv
import numpy as np

def calc_ppr(row):
    return (float(row["RushingYards"]) / 10) + (float(row["RushingTouchdowns"]) * 6) + (float(row["Receptions"])) + (float(row["ReceivingTouchdowns"]) * 6) + (float(row["ReceivingYards"]) / 10)

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
    shared_chars = ['Name', 'Team', 'Position', 'Week', 'Opponent', "Year"]
    # Get rid of the bums
    df_mask = (df_proj['RushingAttempts'] == 0) & (df_proj['RushingYards'] == 0) & (df_proj['RushingYardsPerAttempt'] == 0) & (df_proj['RushingTouchdowns'] == 0) & (df_proj['ReceivingTargets'] == 0) & (df_proj['Receptions'] == 0) & (df_proj['ReceivingYards'] == 0) & (df_proj['ReceivingTouchdowns'] == 0)
    df_proj = df_proj[~df_mask]

    final_dict = make_defense_dict()

    columns = ["TacklesForLoss","Sacks","QuarterbackHits","Interceptions","FumblesRecovered","Safeties","DefensiveTouchdowns","SpecialTeamsTouchdowns","PointsAllowedByDefenseSpecialTeams","DefFantasyPointsPerGame","DefFantasyPoints","DefRank"]
    list_proj = []
    for i, row in df_proj.iterrows():
        new_row = final_dict[row["Year"] - 2000][row["Opponent"]][int(row["Week"])]["row"]
        new_row["DefRank"] = final_dict[row["Year"] - 2000][row["Opponent"]][row["Week"]]["rank"]
        list_proj.append(new_row)

    list_actual = []
    for i, row in df_actual.iterrows():
        new_row = final_dict[row["Year"] - 2000][row["Opponent"]][int(row["Week"])]["row"]
        new_row["DefRank"] = final_dict[row["Year"] - 2000][row["Opponent"]][row["Week"]]["rank"]
        list_actual.append(new_row)


    df_list_proj = pd.DataFrame(list_proj, columns=columns)
    df_list_actual = pd.DataFrame(list_actual, columns=columns)

    new_indicies_proj = [i for i in range(len(df_list_proj))]
    new_indicies_actual = [i for i in range(len(df_list_actual))]

    df_list_proj.index = new_indicies_proj
    df_proj.index = new_indicies_proj

    df_list_actual.index = new_indicies_actual
    df_actual.index = new_indicies_actual

    df_combined_proj = pd.concat([df_proj, df_list_proj], axis=1)
    df_combined_actual = pd.concat([df_actual, df_list_actual], axis=1)

    trimmed_df_proj = df_combined_proj.merge(df_combined_actual[shared_chars], on=shared_chars, how='inner')
    trimmed_df_actual = df_combined_actual.merge(df_combined_proj[shared_chars], on=shared_chars, how='inner')


    trimmed_df_proj["FantasyPoints"] = trimmed_df_proj.apply(calc_ppr, axis=1)
    trimmed_df_actual["FantasyPoints"] = trimmed_df_actual.apply(calc_ppr, axis=1)

    saved_column_proj = trimmed_df_proj.pop('FantasyPoints')
    saved_column_actual = trimmed_df_actual.pop('FantasyPoints')

    trimmed_df_proj["FantasyPoints"] = saved_column_proj
    trimmed_df_actual["FantasyPoints"] = saved_column_actual

    labels_vector = trimmed_df_actual.iloc[:, -1] - trimmed_df_proj.iloc[:, -1]
    trimmed_df_proj.iloc[:, -1] = labels_vector
    ['ProjRushYd', 'ProjRushAtt', 'ProjRushTD', 'ProjRecYd', 'ProjRecCount', 'ProjRecTD', 'DiffPPR1',"TacklesForLoss","Sacks","QuarterbackHits","Interceptions","FumblesRecovered","Safeties","DefensiveTouchdowns","SpecialTeamsTouchdowns","PointsAllowedByDefenseSpecialTeams","DefFantasyPointsPerGame","DefFantasyPoints","DefRank"]
    trimmed_df_proj = trimmed_df_proj.rename(columns={"FantasyPoints": "DiffPPR1",
                                                    "RushingAttempts": 'ProjRushAtt',
                                                    "RushingYards": 'ProjRushYd',
                                                    "RushingTouchdowns": 'ProjRushTD',
                                                    "Receptions": 'ProjRecCount',
                                                    "ReceivingTouchdowns": "ProjRecTD",
                                                    "ReceivingYards": "ProjRecYd"})

    trimmed_df_proj = trimmed_df_proj.drop_duplicates(keep=False)
    trimmed_df_proj.to_csv(output, index=False)

def make_defense_dict():
    # Get the 2012 dictionary
    data_2012 = pd.read_csv("../input_data/DST/DST_2012_season.csv")
    data_2012.iloc[:, 4:] = data_2012.iloc[:, 4:].astype(float) / (17)
    rows_2012 = {}
    rankings_dict = {}
    rankings_dict[13] = {}
    rankings_dict[13][1] = {}
    for i, row in data_2012.iterrows():
        rows_2012[row[2]] = row[6:]
        rankings_dict[13][1][row["Team"]] = i + 1
    # Make the rankings dictionary
    byes = {}
    for i in range(13, 23):
        byes[i] = {}
        if i != 13:
            rankings_dict[i] = {}
        rankings_df = pd.read_csv(f"../input_data/DST_Rankings/rankings20{i}.csv")
        team_scores = {}
        for week in rankings_df.columns[4:-2]:
            week = int(week)
            rankings_dict[i][week + 1] = {}
            teams_list = []
            for j, row in rankings_df.iterrows():
                if (row[str(week)] == "BYE" or row[str(week)] == "-"):
                    byes[i][row["Team"]] = week
                if row["Team"] not in team_scores:
                    team_scores[row["Team"]] = 0
                if row[str(week)] != "BYE" and row[str(week)] != '-':
                    team_scores[row["Team"]] += float(row[str(week)])
                teams_list.append((row["Team"], team_scores[row["Team"]]))
            teams_list = sorted(teams_list, key=lambda x: x[1], reverse=True)
            if week == 1 and i != 13:
                old_rankings_df = pd.read_csv(f"../input_data/DST_Rankings/rankings20{i - 1}.csv")
                rankings_dict[i][1] = {}
                for c, row in old_rankings_df.iterrows():
                    rankings_dict[i][week][row["Team"]] = i + 1

            for z, (team, _) in enumerate(teams_list):
                rankings_dict[i][week + 1][team] = z + 1
        
    # Just build the dictionary, we'll iterate over it later
    # iterate over years
    #construct the dictionary skeleton
    defense_dict = {}
    for i in range(13, 23):
        rankings_path = f"../input_data/DST_Rankings/rankings20{i}.csv"
        data_path = f"../input_data/DST/DST_20{i}.csv"
        rankings_df = pd.read_csv(rankings_path, header=0)
        data_df = pd.read_csv(data_path, header=0)
        defense_dict[i] = {}
        # add each year as a key in the dictionary
        for team in rankings_df["Team"]:
            # add each team to the year dict. In the list, we'll store the past week's data. If the team is on bye, then we store the previous week. If it's week 1, look at last year's data
            defense_dict[i][team] = {}
            list_weeks = list(rankings_df.columns[4:-2])
            list_weeks.extend([18])
            for week in list_weeks:
                week = int(week)
                defense_dict[i][team][week] = {}
                defense_dict[i][team][week]["rank"] = rankings_dict[i][week][team]
            if i == 13:
                defense_dict[i][team][1]["row"] = rows_2012[team]

    #populate the dictionary
    all_first_rows = {}
    for i in range(13, 22):
        all_first_rows[i] = {}
        data = pd.read_csv(f"../input_data/DST/DST_20{i}_season.csv")
        data.iloc[:, 4:] = data.iloc[:, 4:].astype(float) / (17)
        for j, row in data.iterrows():
            all_first_rows[i][row[2]] = row[6:]
            rankings_dict[13][1][row["Team"]] = j + 1

    defense_dict[17]["TB"][1]["row"] = all_first_rows[17 - 1]["TB"]
    defense_dict[17]["TB"][2]["row"] = all_first_rows[17 - 1]["TB"]
    defense_dict[17]["TB"][3]["row"] = all_first_rows[17 - 1]["TB"]
    defense_dict[17]["MIA"][1]["row"] = all_first_rows[17 - 1]["MIA"]
    defense_dict[17]["MIA"][2]["row"] = all_first_rows[17 - 1]["MIA"]
    defense_dict[17]["MIA"][3]["row"] = all_first_rows[17 - 1]["MIA"]
    byes[17]["MIA"] = 1
    byes[17]["TB"] = 1
    for i in range(13, 23):
        rankings_path = f"../input_data/DST_Rankings/rankings20{i}.csv"
        data_path = f"../input_data/DST/DST_20{i}.csv"
        rankings_df = pd.read_csv(rankings_path, header=0)
        data_df = pd.read_csv(data_path, header=0)
        data_df = data_df.sort_values(by=["Name","Week"])
        for _, row in data_df.iterrows():
            if (i == 17 and row["Team"] == "TB" and row["Week"] == 2) or (i == 17 and row["Team"] == "MIA" and row["Week"] == 2):
                continue
            elif int(row["Week"]) + 1 == 19:
                continue
            if int(row["Week"]) == 1 and i != 13:
                defense_dict[i][row["Team"]][1]["row"] = all_first_rows[i - 1][team]
                defense_dict[i][str(row["Team"])][int(row["Week"]) + 1]["row"] = row[6:]
            elif int(row["Week"]) == byes[i][row["Team"]] - 1:
                defense_dict[i][str(row["Team"])][int(row["Week"]) + 2]["row"] = defense_dict[i][str(row["Team"])][int(row["Week"])]["row"]
            else:
                defense_dict[i][str(row["Team"])][int(row["Week"]) + 1]["row"] = row[6:]

            defense_dict[i][str(row["Team"])][int(row["Week"]) + 1]["rank"] = rankings_dict[i][int(row["Week"])][row["Team"]]

    return defense_dict
    

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