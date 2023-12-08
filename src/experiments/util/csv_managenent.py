import pandas as pd
import sys

def sort_csv(position, proj_or_real):
    input_file = "../../input_data/" + position.upper() + "s/" + position + "_13-22_" + proj_or_real + ".csv"

    # read in csv to pandas dataframe
    df = pd.read_csv(input_file)

    # sort by the following columns
    columns_to_sort_by = ["Name", "Year", "Week"]

    # Sort the DataFrame by the specified columns
    df_sorted = df.sort_values(by=columns_to_sort_by)
    
    # desired output file location
    output_file = "../../input_data/" + position.upper() + "s/" + position + "_13-22_" + proj_or_real + "_sorted.csv"

    # Save the sorted DataFrame to a new CSV file
    df_sorted.to_csv(output_file, index=False)
    return

def add_year(position, proj_or_real):
    input_file = "../../input_data/" + position.upper() + "s/" + position + "_13-22_" + proj_or_real + ".csv"

    # read in sv to pandas df
    df = pd.read_csv(input_file)

    mask = (df['Rank'] == 1)

    # Create a mask for rows where the last occurrence of 'Column1' was at least two rows above
    mask_last_occurrence = mask & (~mask.shift(fill_value=False)) & (~mask.shift(-1, fill_value=False))

    # Create the 'NewColumn' based on conditions
    df['Year'] = 2013 + mask_last_occurrence.cumsum() - 1

    print(df['Name'])
    print(df['Year'])

    #output_file = "../../input_data/" + position.upper() + "s/" + position + "_13-22_" + proj_or_real + "_year.csv"

    #df.to_csv(output_file, index=False)
    #return

def main():
    args = sys.argv[1:]
    if len(args) != 3:
            print("Too many or too few args.")
            return
    if args[0] == "sort":
        sort_csv(args[1], args[2])
    elif args[0] == "year":
        add_year(args[1], args[2])
    else:
         print("Not valid command")


if __name__ == "__main__":
    main()