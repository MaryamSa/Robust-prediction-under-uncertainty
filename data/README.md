# Data

The pipeline downloads the fixed **Kepler Q1–Q17 Data Release 25 Objects of Interest table** from the NASA Exoplanet Archive through its TAP service.

- Table: `q1_q17_dr25_koi`
- Signals: 8,054
- Pipeline dispositions: 4,034 `CANDIDATE`, 4,020 `FALSE POSITIVE`
- Dataset DOI: [10.26133/NEA5](https://doi.org/10.26133/NEA5)
- [NASA table documentation](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)
- [NASA column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)

The exact query is constructed in `src/kepler_uncertainty/data.py`. Downloaded data are cached in `data/raw/` and ignored by Git.

## Target

`koi_pdisposition` is the Kepler pipeline disposition:

- `CANDIDATE` becomes 1;
- `FALSE POSITIVE` becomes 0.

A candidate is **not** a confirmed planet. This project estimates similarity to the DR25 pipeline label; it does not validate exoplanets.

## Model inputs

The 11 inputs describe orbital period, impact parameter, transit duration and depth, candidate radius, equilibrium temperature, transit-model signal-to-noise, stellar temperature, surface gravity and radius, and Kepler magnitude.

Positive quantities are transformed with `log(1 + x)` to reduce extreme scale differences. Missing values are replaced with the training median inside each model fold, so information from held-out data cannot leak into preprocessing.

## Leakage controls

The download deliberately excludes fields that directly encode or explain the disposition: `koi_score`, all four `koi_fpflag_*` flags, `koi_disposition`, and `koi_comment`. Identifiers are retained for grouping and traceability but are not model inputs.

## Measurement uncertainties

Quoted upper and lower errors for eight inputs are retained for Monte Carlo propagation. The project approximates one standard deviation by the mean absolute upper/lower error and draws independent normal perturbations. This is transparent but simplified; correlations, systematic errors, label uncertainty, and model uncertainty are not included.

