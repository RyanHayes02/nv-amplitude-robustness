# Threshold analysis: quick start

Put `security_metrics.py` and `nv_security_engine.py` beside `run_analysis.py` in the repository root. No changes to the original benchmark are needed. The helper engine is extracted from commit d093f51 without changing its physics; it avoids running the full benchmark or optimizer on import.

From a terminal in that folder:

```bash
python -m pip install -r requirements.txt
python security_metrics.py
```

In a notebook, with both files in the current folder:

```python
%run security_metrics.py
```

The script prints progress. Open the saved PNG after it finishes; it does not display a plot window. It uses the selective-drive, single-nuclear-state scenario, the original 25 fields, 20 us DQ dephasing, 6 us wait, stored OC pulse, and frozen nominal calibration. No hardware-scenario or single-tone claim follows from this run.

## Outputs

Each run makes a new timestamped folder under `security_results/`:

- `per_field_errors.csv`: true field, signal, estimated field, signed error, validity.
- `threshold_metrics.csv`: counts and fractions for thresholds 1, 5, 10, 50 nT and common amplitude scales 0.6, 0.9, 1.1, 1.5.
- `threshold_fractions.pdf` / `.png`: paper figure.
- `threshold_*_nT.tex`: tables (fractions 0–1).
- `maximum_bias_checks.csv`: compare maximum errors against the original paper table. A maximum over only valid points is separately labeled and is not a full-grid maximum.
- `nominal_checks.csv` and `settings.json`: checks and reproducibility metadata.

The denominator is ALL tested fields. A point counts only if the estimate is finite AND absolute error is strictly greater than the threshold. Invalid points do not count as successful valid-output corruption, but their separate fractions must be reported; invalid does not mean an attack was detected or harmless. Thresholds are illustrative, not application requirements. Fractions on this deterministic grid are not probabilities over an operational field distribution. Results use expected signals, not simulated finite-shot readings.

Optional denser grid (longer run):

```bash
python security_metrics.py --points 101
```

## GitHub

Add the two Python files and this README to the repository root. Run the script locally, then upload the completed timestamped output folder under `security_results/`. Keep the settings and raw per-field CSV along with the figure. Do not upload `__pycache__`.

Before citing results, compare `maximum_bias_checks.csv` with `reference_results/table1_bias.csv`, review nominal checks, and confirm every expected case completed. The engine and extraction provenance are recorded in `nv_security_engine.py`. This script adds analysis; it does not modify or publish the original repository automatically.

The package includes a completed 25-field run in security_results/20261028T183724Z. All 16 maxima match the rounded baseline within 0.005 nT. Reduced smoke testing and strict-threshold/invalid-denominator checks also passed.
