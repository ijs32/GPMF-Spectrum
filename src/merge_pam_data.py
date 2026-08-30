import pandas as pd
import numpy as np
from datetime import datetime as dt
import time

# Source - https://stackoverflow.com/a/6451892
# Posted by ninjagecko, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-24, License - CC BY-SA 3.0
def to_year_fraction(date):
    def since_epoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = since_epoch

    year = date.year
    start_of_this_year = dt(year=year, month=1, day=1)
    start_of_next_year = dt(year=year+1, month=1, day=1)

    year_elapsed = s(date) - s(start_of_this_year)
    year_duration = s(start_of_next_year) - s(start_of_this_year)
    fraction = year_elapsed/year_duration

    return date.year + fraction


def load_pam() -> pd.DataFrame:
    df = pd.read_csv(
        "./data/pamela/txt/File_content_index.txt",
        sep=" ; "
    )

    df = df.rename(columns={'Start Date': 'Start_Date', 'Stop Date': 'Stop_Date'})

    df["Start_Date"] = pd.to_datetime(df["Start_Date"])
    df["Stop_Date"]  = pd.to_datetime(df["Stop_Date"])

    file_path = "./data/pamela/txt/"
    df_list   = []

    for row in df.itertuples():
        file = (row.Filename).split(".")[0] + ".txt"

        decimal_start_year = to_year_fraction(row.Start_Date) 
        decimal_stop_year  = to_year_fraction(row.Stop_Date)

        avg_year = (decimal_start_year + decimal_stop_year) / 2

        pam = pd.read_csv(
            file_path + file,
            sep=r"\s+",
            comment="#",
            header=None,
            engine='python',
            names=["E", "E_err_low", "E_err_high", "flux", "flux_err_low", "flux_err_high"],
        )
        pam["date"] = avg_year

        df_list.append(pam)

    final =  pd.concat(df_list, ignore_index=True)
    return final[final["E"] <= 1]


def main():
    pam = load_pam()

    pam.to_csv("./data/pamela/pam_combined.csv", index=False)


if __name__ == "__main__":
    main()
