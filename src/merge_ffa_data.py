import pandas as pd
import numpy as np
from datetime import datetime as dt
from pathlib import Path

def load_ffa() -> pd.DataFrame:

    file_path = "./data/ffa/"
    df_list   = []

    for f in Path(file_path).iterdir():
        filename = f.name
        date = f.stem.split("-")[1]

        ffa = pd.read_csv(
            file_path + filename,
            sep=r"\s+",
            comment="#",
            header=1,
            engine='python',
            names=["E", "flux"],
        )
        ffa["date"] = date
        ffa["E"] = ffa["E"] / 1000 # convert to GeV
        ffa["flux"] = ffa["flux"] * 1000

        df_list.append(ffa)

    final =  pd.concat(df_list, ignore_index=True)
    return final[final["E"] <= 1]


def main():
    ffa = load_ffa()

    ffa.to_csv("./data/ffa/ffa_combined.csv", index=False)


if __name__ == "__main__":
    main()
