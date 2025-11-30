# Cepheid Variable Period-Luminosity Relation

## Project Overview

This project implements a complete fitting pipeline to determine the Period-Luminosity (P-L) relation for Cepheid variable stars using real observational data from the OGLE survey. The P-L relation is a fundamental tool in extragalactic astronomy, serving as a "standard candle" for measuring cosmic distances.

## Scientific Background

Cepheid variables are pulsating stars whose brightness oscillates with regular periods ranging from 1 to 100 days. The critical discovery by Henrietta Leavitt (1912) established that the pulsation period directly correlates with the star's intrinsic luminosity: longer periods correspond to more luminous stars. This relationship enables astronomers to:

1. Measure the pulsation period from light curves
2. Determine the absolute magnitude using the P-L relation
3. Compare to the observed apparent magnitude
4. Calculate the distance using the distance modulus

This method was instrumental in Edwin Hubble's 1920s measurements of galaxy distances and the discovery of the universe's expansion.

## Methodology

The project implements a custom fitting pipeline using `scipy.optimize.curve_fit` to fit the linear relation:

```
magnitude = a × log₁₀(period) + b
```

where `a` is the slope parameter and `b` is the intercept (which includes the distance modulus for apparent magnitudes).

### Key Steps

1. **Data Acquisition**: Download Cepheid catalog data from the OGLE-III survey via VizieR using astroquery
2. **Data Exploration**: Clean and analyze the dataset, checking for data quality issues
3. **Model Fitting**: Implement custom model function and fit using scipy.optimize
4. **Statistical Analysis**: Calculate chi-squared, reduced chi-squared, R², and residual analysis
5. **Visualization**: Create publication-quality plots showing data, fitted model, and residuals
6. **Distance Estimation**: Demonstrate how the fitted relation is used for distance measurements

## Requirements

- Python 3.7 or higher
- Internet connection (for data download)
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository or download the project files
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Open `main.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially
3. The notebook will:
   - Download Cepheid data from VizieR (requires internet connection)
   - Save data to `ogle_lmc_cepheids.csv`
   - Fit the Period-Luminosity relation
   - Generate statistical analysis
   - Create visualization saved as `cepheids_pl_relation.png`

## Data Source

The project uses data from the OGLE-III survey (Optical Gravitational Lensing Experiment), specifically catalog `J/AcA/58/163` containing VI light curves of classical Cepheids in the Large Magellanic Cloud (LMC). All stars in this catalog are from the same galaxy, making them ideal for fitting the P-L relation using apparent magnitudes.

## Output Files

- `ogle_lmc_cepheids.csv`: Cleaned Cepheid data (StarID, Period, I_mag)
- `cepheids_pl_relation.png`: Publication-quality figure showing the P-L relation and residuals


