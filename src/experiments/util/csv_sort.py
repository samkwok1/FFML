import pandas as pd
import sys

def sort_csv(position, proj_or_real):
    input_file = "../../input_data/" + position.upper() + "s/" + position + "_13-22_" + proj_or_real + ".csv"

    # read in csv to pandas dataframe
    df = pd.read_csv(input_file)

    # sort by the following columns
    columns_to_sort_by = ["Name", "Week"]

    # Sort the DataFrame by the specified columns
    df_sorted = df.sort_values(by=columns_to_sort_by)
    
    # desired output file location
    output_file = "../../input_data/" + position.upper() + "s/" + position + "_13-22_" + proj_or_real + "_sorted.csv"

    # Save the sorted DataFrame to a new CSV file
    df_sorted.to_csv(output_file, index=False)
    return

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Too many or too few args.")
        return
    sort_csv(args[0], args[1])


if __name__ == "__main__":
    main()