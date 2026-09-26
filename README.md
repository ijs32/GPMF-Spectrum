# GETTING STARTED

## SIMPLE STARTUP
To ensure the code will run on whatever system you are using, I recommend the using the uv virtual environment. This project comes with a pyprotject.toml file, if you have some other preferred method, it should be simple enough to download the necessary packages.

### UV INSTALL
Run the following command to install uv.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## DATA (OPTIONAL)
All data required to run the model and generate plots is include in the `./src/data` directory. However, if you like to
pull the data yourself, you can find it [here (PAMELA)](https://example.com)  and [here (FFA)](https://example.com)

If you do decide to pull the data yourself, you will need to make the following changes to the original PAMELA data for my script in `main.py` to work:
- deleted extra time_min= and time_max= off the end of data so column header count matches column count
- deleted first row

## RUN
To interact with the model, run all commands from within the `./src` directory:

```bash
cd src
```

### REGENERATE MODEL (OPTIONAL)
This repo comes with a model presaved under `./src/data/preds/mf_preds.npz`. However if you wish to generate the model from scratch, run the following:

```bash
uv run generate_prediction_data.py
```

On your first usage of the uv command, uv will create a virtual environment and install all the necessary dependencies. This may take a moment, as will running generate_prediction_data.py.

### PLOTS
In order to generate plots using the model, run:

```bash
uv run plots.py
```

Doing so will generate a rainbow plot of the spectra across 6 epochs, as well as a 6 panel plot of the model versus the FFA. This code could easily be modified to your liking.

### TESTING
To validate our model's accuracy, we do simple 5 fold validation holding out 20% of the PAMELA dataset as a test set. The model scores upwards of ~95% accuracy on a given test set. To validate the model yourself, run:

```bash
uv run test_gpmf_spectrum.py
```

This may take some time.