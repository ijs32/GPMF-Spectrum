import numpy as np
from sklearn.model_selection import KFold

from shared.emukitGP import EmuKitGP


def test_model(pam):
    # transform: log10 energy, natural-log flux, standardized
    E    = pam[:, 0]
    date = pam[:, 6]
    flux = pam[:, 3]
    ferr = pam[:, 4]

    X = np.column_stack([np.log10(E), date])

    ylog = np.log(flux)
    y_mu, y_sd = ylog.mean(), ylog.std()
    y    = (ylog - y_mu) / y_sd
    yerr = (ferr / flux) / y_sd

    xmax = X.max(axis=0)
    xmin = X.min(axis=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds, truths = [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        y_err_tr   = yerr[train_idx]

        model = EmuKitPam(X_tr, y_tr, y_err_tr, xmax, xmin)
        mean = model.evaluate(X_te).ravel()

        preds.append(mean)
        truths.append(y_te)

    preds  = np.concatenate(preds)
    truths = np.concatenate(truths)

    rmse_std = np.sqrt(np.mean((preds - truths) ** 2))

    # back to real flux for an interpretable number
    flux_pred = np.exp(preds  * y_sd + y_mu)
    flux_true = np.exp(truths * y_sd + y_mu)
    rel_err = np.mean(np.abs(flux_pred - flux_true) / flux_true)

    print(f"CV RMSE (standardized log-flux): {rmse_std:.4f}")
    print(f"CV mean relative error on flux:  {rel_err:.2%}")


def main():
    # E,E_err_low,E_err_high,flux,flux_err_low,flux_err_high,date
    pam = np.loadtxt('./data/pamela/pam_combined.csv', delimiter=',', skiprows=1)

    test_model(pam)


if __name__ == "__main__":
    main()