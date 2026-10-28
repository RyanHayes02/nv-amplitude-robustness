# NV DQ Ramsey control under amplitude manipulation

Simulation code comparing hard pulses, BB1, fractional STIRAP, and a stored time-matched optimized-control pulse in NV-center double-quantum Ramsey magnetometry.

The benchmark measures field-estimation bias under a frozen honest calibration, nominal local sensitivity, and an illustrative bias–noise crossover. It also evaluates independent-tone attenuation, cross-driving, static nuclear-state mixtures, and selected numerical-reference and convergence checks.

## Quick start

Use Python 3.12 or 3.13 in an isolated environment. From the repository directory:

```bash
python -m pip install -r requirements.txt
python run_analysis.py
```

For the notebook interface:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab nv_control_analysis.ipynb
```

Run notebook cells from top to bottom. The notebook and script contain the same analysis code; use one entry point per run. For unattended or headless execution, set the Matplotlib backend to `Agg` using the `MPLBACKEND` environment variable. Numerical work is performed on the CPU. No hardware access, credentials, or network data retrieval is required by the analysis.

The reported reference run used Python 3.13.2, NumPy 2.2.3, SciPy 1.15.2, pandas 2.2.3, and Matplotlib 3.10.7. `requirements-reference.txt` pins these numerical packages; the general requirements allow compatible versions and do not promise bit-for-bit reproducibility. Runtime depends on CPU and package versions; cross-driving calculations dominate the cost.

## Files

| Path | Purpose |
| --- | --- |
| `nv_control_analysis.ipynb` | Notebook interface for the baseline benchmark |
| `run_analysis.py` | Python entry point for the baseline benchmark |
| `reference_results/` | Archived baseline results |
| `security_metrics.py` | Threshold analysis of valid but biased readings |
| `nv_security_engine.py` | Supporting engine for threshold analysis |
| `README_security_metrics.md` | Instructions for running and interpreting threshold analysis |
| `security_results/` | Archived threshold-analysis results |
| `validate_revision.py` | Additional grid, inverse, hardware, and independent-solver checks |
| `summarize_revision.py` | Aggregates the additional validation results |
| `revision_results/verified/` | Archived validation results and settings |
| `pulse_specification.tex` | Detailed pulse envelopes and phase conventions |
| `CITATION.cff` | Software citation metadata |
| `requirements*.txt` | Numerical and notebook dependencies |

## Configuration

Settings are defined near the start of the analysis and in the configuration dataclasses. Defaults include a 6 microsecond free-evolution interval, 25 residual fields from −0.2 to +0.2 microtesla, and common amplitude factors from 0.5 to 1.5. The default outer time step is 0.5 ns. See `run_settings.json` for the complete configuration of each completed run.

The optimized pulse coefficients are stored in the code. `RUN_OC_DESIGN = True` repeats the seeded six-start design and substantially increases runtime. Its objective is a bright-state rotation over an amplitude grid, not a field-aware global optimum. Re-optimizing can change the reported coefficients and results.

`RUN_SLOW_REFERENCE_51 = True` enables an additional independent STIRAP reference at 51 mT. It was **not enabled in the reported reference run**. Selected 51 mT bias-convergence checks are included by default.

## Outputs

Each run writes `paper_results/<UTC timestamp>/` and creates `NV_robust_control_results_<timestamp>.zip`. Outputs include:

- Field-bias sweeps, sensitivity/crossover tables, and hardware-scenario tables.
- PNG and vector PDF figures.
- Baseline reproduction, independent-reference, and numerical-convergence tables.
- Configuration, package versions, stored pulse coefficients, validation status, run log, and checksum manifest.

Read `run_status.json` and the individual check tables before interpreting results. The checksum manifest describes the files included in the exported run. Save the executed notebook separately if its displayed outputs are needed. Generated results and archives are ignored by Git; the supplied `reference_results/` directory is retained.

## Model and interpretation

The state ordering is `|+1>, |0>, |-1>`. Units are microseconds, microtesla, and angular frequencies in radians per microsecond. The electronic gyromagnetic ratio is `2*pi*0.028 rad/(microsecond*microtesla)`.

The simulation uses a three-level rotating-wave model, quasistatic Gaussian field averaging, and static secular nitrogen nuclear-state offsets. Cross-driving uses continuous absolute microwave phase. The 51 mT scenario assumes a 95% majority nuclear population; that is an input, not a measured preparation efficiency. It does not simulate optical nuclear preparation or a full electron–nuclear Hamiltonian.

Common amplitude scaling preserves the two-tone ratio but changes the adiabatic gap. Independent-tone scaling changes the dark state. Neither fractional STIRAP nor any comparator is assumed immune to arbitrary waveform, phase, frequency, or timing manipulation. Real amplifier compression, AM-to-PM conversion, carrier counter-rotating terms, and a measured noise spectrum are outside the model.

Calibration is frozen before manipulation. Nonmonotone calibration branches are rejected, and attacked readings outside the inverse-calibration domain are not clipped or extrapolated. A missing maximum means at least one field estimate is invalid under that case.

Bias maxima are over the sampled field grid, not certified continuous worst-case bounds. Sensitivity includes modeled readout noise and cycle time. The crossover is an illustrative comparison of bias and local statistical noise, not an attack-detection time. Numerical agreement is limited to the tested cases and cannot validate omitted physics. No experimental validation, malicious-intent attribution, or general security guarantee is claimed.

## Citation

If you use this code or the accompanying results, please cite this repository using the metadata in CITATION.cff and identify the commit used.

The manuscript cites commit 8c4f1ef97636ac4ae213a30f119fb6ce2eb1a584, which includes the baseline results, threshold analysis, and additional numerical validation.

## AI assistance

OpenAI's ChatGPT/Codex and Anthropic's Claude assisted with code development, debugging, and manuscript preparation, including drafting, editing, and formatting. The authors take responsibility for the final code, analysis, and manuscript.

## License
A software license has not yet been selected. The code and results are publicly available for inspection and reproducibility; reuse permissions will be specified when a license is added.

A license has not been selected. Add an appropriate license before describing the repository as open-source or granting reuse permissions. Confirm the author and contributor metadata before public release.
