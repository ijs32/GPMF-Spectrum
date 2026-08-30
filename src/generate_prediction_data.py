import numpy as np

from shared.emukitMFGP import EmukitMFGP

import logging
from datetime import datetime

# Configure logging to save to a file
logging.basicConfig(
    filename='./data/generate_prediction_data.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

pam = np.loadtxt('./data/pamela/pam_combined.csv', delimiter=',', skiprows=1)
ffa = np.loadtxt('./data/ffa/ffa_combined.csv', delimiter=',', skiprows=1)

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

start_fit = datetime.now()
start_fit_time = start_fit.time()
logging.info(f"Starting fit of model at {start_fit_time}")

mf = EmukitMFGP(X_lo, y_lo, X_hi, y_hi, xmin, xmax)

end_fit = datetime.now()
end_fit_time = end_fit.time()
logging.info(f"fit of model complete at {end_fit_time}")
logging.info(f"duration: {(end_fit - start_fit)}")

start_pred = datetime.now()
start_pred_time = start_pred.time()
logging.info(f"starting predictions at {start_pred_time}")

mean, sd = mf.evaluate_uncertainty(X_lo)
preds, sigma = mean.ravel(), sd.ravel()

end_pred = datetime.now()
end_pred_time = end_pred.time()
logging.info(f"finished predictions at {end_pred_time}")
logging.info(f"duration: {(end_pred - start_pred)}")

np.savez('./data/preds/mf_preds.npz',
         E=E_lo, date=date_lo,
         preds=preds, sigma=sigma,
         y_mu=y_mu, y_sd=y_sd)