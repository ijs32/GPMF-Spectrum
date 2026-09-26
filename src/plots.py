import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from datetime import datetime, timedelta

def decimal_year_to_datetime(decimal_year):
    year = int(decimal_year)
    remainder = decimal_year - year
    
    start_of_year = datetime(year, 1, 1)
    start_of_next_year = datetime(year + 1, 1, 1)
    year_duration_sec = (start_of_next_year - start_of_year).total_seconds()
    
    dt = start_of_year + timedelta(seconds=remainder * year_duration_sec)
    return dt.strftime("%d %b %Y")


def generate_plots(pam, ffa, preds, sigma, dates, y_mu, y_sd, emax=10.0,
                   out='./plots/spectra_panels_1.png'):
    """preds, sigma: standardized-log-flux mean and sd, row-aligned with ffa."""
    E_p, flux_p, ferr_p, date_p = pam[:, 0], pam[:, 3], pam[:, 4], pam[:, 6]
    E_f, flux_f, date_f = ffa[:, 0], ffa[:, 1], ffa[:, 2]

    lo  = np.exp((preds - 2*sigma) * y_sd + y_mu)
    hi  = np.exp((preds + 2*sigma) * y_sd + y_mu)
    mid = np.exp(preds * y_sd + y_mu)

    ncol = 3
    nrow = int(np.ceil(len(dates) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.4*nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    ffa_dates = np.unique(date_f)
    for ax, d in zip(axes, dates):
        d_format = decimal_year_to_datetime(d)
        d_f = ffa_dates[np.argmin(np.abs(ffa_dates - d))]
        m = (date_f == d_f) & (E_f <= emax)
        o = np.argsort(E_f[m])

        ax.fill_between(E_f[m][o], lo[m][o], hi[m][o],
                        color='tab:red', alpha=0.2, linewidth=0, zorder=1)
        ax.plot(E_f[m][o], mid[m][o], color='tab:red', lw=1.3,
                label='GPR', zorder=3)
        ax.plot(E_f[m][o], flux_f[m][o], color='tab:green', lw=1.0,
                ls='--', label='FFA', zorder=2)

        mp = date_p == d
        op = np.argsort(E_p[mp])
        ax.errorbar(E_p[mp][op], flux_p[mp][op], yerr=ferr_p[mp][op],
                    color='k', marker='o', markersize=2.5, linestyle='none',
                    elinewidth=0.7, capsize=0, label='PAMELA', zorder=4)

        ax.axvspan(E_f[m].min(), E_p.min(), color='0.85', alpha=0.5, zorder=0)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.margins(x=0)
        ax.set_title(f'{d_format}', fontsize=10)
        ax.grid(True, which='both', alpha=0.2)

    for ax in axes[len(dates):]:
        ax.set_visible(False)

    axes[0].legend(fontsize=8, loc='lower right')
    fig.supxlabel('Energy [GeV]')
    fig.supylabel(r'Flux [m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]')
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_rainbow_preds(E, date, preds, sigma, y_mu, y_sd, epochs, pam,
                       emax=1.0, band=True, cmap='turbo',
                       out='./plots/rainbow_preds_1.png'):
    keep = E <= emax
    E, date, preds, sigma = E[keep], date[keep], preds[keep], sigma[keep]

    mid = np.exp(preds * y_sd + y_mu)
    lo  = np.exp((preds - 2*sigma) * y_sd + y_mu)
    hi  = np.exp((preds + 2*sigma) * y_sd + y_mu)

    epochs = np.sort(np.asarray(epochs))
    colors = mpl.colormaps[cmap](np.linspace(0.05, 0.95, len(epochs)))
    avail_pred = np.unique(date)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    E_p, flux_p, ferr_p, date_p = pam[:, 0], pam[:, 3], pam[:, 4], pam[:, 6]
    avail_pam = np.unique(date_p)
    ax.axvspan(E.min(), E_p.min(), color='0.9', zorder=0)

    for c, d_req in zip(colors, epochs):
        dd = avail_pred[np.argmin(np.abs(avail_pred - d_req))]
        m = date == dd

        date_formatted = decimal_year_to_datetime(dd)

        o = np.argsort(E[m])
        if band:
            ax.fill_between(E[m][o], lo[m][o], hi[m][o],
                            color=c, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(E[m][o], mid[m][o], color=c, linewidth=1.6,
                label=f'{date_formatted}', zorder=2)

        dp = avail_pam[np.argmin(np.abs(avail_pam - d_req))]
        mp = date_p == dp
        op = np.argsort(E_p[mp])
        ax.errorbar(E_p[mp][op], flux_p[mp][op], yerr=ferr_p[mp][op],
                    color=c, marker='o', markersize=3.5, linestyle=':',
                    markeredgecolor='k', markeredgewidth=0.4,
                    elinewidth=0.7, capsize=0, zorder=3)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.margins(x=0)
    ax.set_xlabel('Energy [GeV]')
    ax.set_ylabel(r'Flux [m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]')
    ax.set_title(r'GPR Model Fit of H$^+$ spectra')
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(title='Date', fontsize=9, title_fontsize=9, loc='lower right')

    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)


pam = np.loadtxt('./data/pamela/pam_combined.csv', delimiter=',', skiprows=1)
ffa = np.loadtxt('./data/ffa/ffa_combined.csv', delimiter=',', skiprows=1)

E_hi, date_hi, flux_hi = pam[:,0], pam[:,6], pam[:,3]
X_hi = np.column_stack([np.log10(E_hi), date_hi])

E_lo, date_lo, flux_lo = ffa[:,0], ffa[:,2], ffa[:,1]
X_lo = np.column_stack([np.log10(E_lo), date_lo])

d = np.load('./data/preds/mf_preds.npz')
E, date_pred = d['E'], d['date']
preds, sigma = d['preds'], d['sigma']
y_mu, y_sd = float(d['y_mu']), float(d['y_sd'])

dates_unique = np.unique(date_hi)

dates = [dates_unique[i*6] for i in range(6)]

generate_plots(pam, ffa, preds, sigma, dates, y_mu, y_sd)

plot_rainbow_preds(E, date_pred, preds, sigma,
                   y_mu, y_sd, dates, pam)

