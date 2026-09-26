from pathlib import Path
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


def generate_plots(pam,ffa,date):

    fig, ax = plt.subplots()

    ax.plot(pam['E'], pam['flux'], marker='o', markeredgewidth=0.5, markersize=2, linestyle='none', label='Pam Spectrum')
    ax.plot(ffa['E'], ffa['flux'], marker='s', markeredgewidth=0.5, markersize=2, linestyle='none', label='FFA Spectrum')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Energy [GeV]')
    ax.set_ylabel(r'Flux [m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]')
    ax.set_title('Spectra' + date)
    ax.legend()

    fig.savefig('./plots/spectra' + date + '.png', dpi=300, bbox_inches='tight')


def unit_correct_ffa(ffa):

    ffa['E']    = ffa['E'].apply(lambda x: x / 1000)
    ffa['flux'] = ffa['flux'].apply(lambda x: x * 1000)

    return ffa


def load_ffa(file):
    file_path = Path(file)
    
    ffa = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=1,
        engine='python',
        names=["E", "flux"]
    )
    return ffa[ffa["E"] <= 1000]


def load_pam(date):
    df = pd.read_csv(
        "./data/pamela/txt/File_content_index.txt",
        sep=" ; ",
        engine='python'
    )

    df["Start Date"] = pd.to_datetime(df["Start Date"])
    target_date = pd.to_datetime(date)

    nearest_idx = (df['Start Date'] - target_date).abs().idxmin()

    file = (df.loc[nearest_idx, 'Filename']).split(".")[0] + ".txt"
    file_path = "./data/pamela/txt/"
    
    pam = pd.read_csv(
        file_path + file,
        sep=r"\s+",
        comment="#",
        header=None,
        engine='python',
        names=["E", "E_err_low", "E_err_high", "flux", "flux_err_low", "flux_err_high"],
    )
    return pam[pam["E"] <= 1]


def main():
    for file in Path("./data/ffa").iterdir():
        date = file.stem.split("-")[1]
        datetime = dt.date(int(date),1,1)

        ffa = load_ffa(file)
        ffa_corrected = unit_correct_ffa(ffa)
        pam = load_pam(datetime)

        generate_plots(pam,ffa_corrected,date)

if __name__ == "__main__":
    main()
