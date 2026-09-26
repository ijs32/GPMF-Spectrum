import numpy as np
from sklearn.model_selection import KFold

from shared.emukitMFGP import EmukitMFGP
from shared.emukitGP import EmuKitGP


def test_model(pam, ffa):
    E_hi, date_hi, flux_hi = pam[:,0], pam[:,6], pam[:,3]
    X_hi = np.column_stack([np.log10(E_hi), date_hi])

    E_lo, date_lo, flux_lo = ffa[:,0], ffa[:,2], ffa[:,1]
    X_lo = np.column_stack([np.log10(E_lo), date_lo])

    ylog_hi, ylog_lo = np.log(flux_hi), np.log(flux_lo)
    allx = np.concatenate([ylog_lo, ylog_hi])
    y_mu, y_sd = allx.mean(), allx.std()

    y_hi = (ylog_hi - y_mu) / y_sd
    y_lo = (ylog_lo - y_mu) / y_sd

    X_all = np.vstack([X_lo, X_hi])
    xmin, xmax = X_all.min(axis=0), X_all.max(axis=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds_mf, preds_sf, truths = [], [], []

    for train_idx, test_idx in kf.split(X_hi):
        Xh_tr, Xh_te = X_hi[train_idx], X_hi[test_idx]
        yh_tr, yh_te = y_hi[train_idx], y_hi[test_idx]

        mf = EmukitMFGP(X_lo, y_lo, Xh_tr, yh_tr, xmin, xmax)
        sf = EmuKitGP(Xh_tr, yh_tr, xmax, xmin)     # same fold, no low-fi data

        preds_mf.append(mf.evaluate(Xh_te).ravel())
        preds_sf.append(sf.evaluate(Xh_te).ravel())
        truths.append(yh_te)

    truths = np.concatenate(truths)
    for name, p in [("multi-fidelity", np.concatenate(preds_mf)),
                    ("single-fidelity", np.concatenate(preds_sf))]:
        rmse = np.sqrt(np.mean((p - truths)**2))
        fp, ft = np.exp(p*y_sd + y_mu), np.exp(truths*y_sd + y_mu)
        print(f"{name:16s}  RMSE {rmse:.4f}   rel err {np.mean(np.abs(fp-ft)/ft):.2%}")


def main():
    # E,E_err_low,E_err_high,flux,flux_err_low,flux_err_high,date
    pam = np.loadtxt('./data/pamela/pam_combined.csv', delimiter=',', skiprows=1)
    ffa = np.loadtxt('./data/ffa/ffa_combined.csv', delimiter=',', skiprows=1)
    
    test_model(pam, ffa)


if __name__ == "__main__":
    main()