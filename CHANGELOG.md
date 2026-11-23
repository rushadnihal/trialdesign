# Changelog

All notable changes to this project. Versions follow [Semantic Versioning](https://semver.org/).

## 0.2.0 (unreleased)

First release on PyPI, as `gsdtrial`. The import (`import trialdesign`) and the command
(`trialdesign`) keep their names. v0.1 config files and Python calls keep working unchanged.

### Added

- **Non-inferiority and superiority-by-a-margin designs** for normal, binary and time-to-event
  endpoints, fixed and group-sequential: `--delta0` (effect under H0, treatment minus
  control) and `--hr0` (hazard ratio under H0). Binary designs use the Farrington-Manning
  restricted MLE under H0.
- **Survival accrual, follow-up and dropout** (Lachin and Foulkes 1986): uniform or piecewise
  accrual, exponential events and dropout. Give two of accrual time, accrual rate and
  follow-up and the third is solved. Reports patients, study duration and the calendar month
  and patients enrolled at each look. New `[accrual]` table and `--median-control`,
  `--accrual-time`, `--accrual-rate`, `--followup`, `--dropout`, `--dropout-time`.
- **Promising-zone sample size re-estimation** (`trialdesign ssr`, `[adaptation]` table):
  conditional power at the interim, Mehta-Pocock promising zone with a cap, and the
  Cui-Hung-Wang weighted test. Exact operating characteristics by one-dimensional
  integration, a simulated check, the naive pooled test for comparison, and the rule applied
  to an observed interim z (`--z1`). Normal and binary endpoints.
- **Spending functions**: Hwang-Shih-DeCani (`hsd:<gamma>`), Kim-DeMets (`kd:<rho>`) and
  user-specified cumulative fractions (`user:<f1>,...,1`), for efficacy and futility.
- **Binding futility**, opt-in with `--binding-futility` or `binding_futility = true`. The
  output also shows the type I error if futility is ignored.
- Report sections and plots for all of the above; R reference scripts and JSON for every new
  number (rpact 3.4.0).
- Release workflow: build, TestPyPI, then PyPI with Trusted Publishing on a `v*` tag.

### Changed

- The PyPI distribution is named `gsdtrial`, because `trialdesign` on PyPI belongs to another
  project. That project also installs a `trialdesign` import, so don't install both in one
  environment.
- The GitHub repository is renamed from `trialdesign-v0.1.0` to `trialdesign`; old links
  redirect.
- Binding futility used to be refused in config files; it is now supported.

## 0.1.0 (2026-11-21)

- Fixed-design sample size for normal, binary and time-to-event endpoints.
- Lan-DeMets O'Brien-Fleming-type and Pocock-type group-sequential boundaries with
  non-binding futility, exact operating characteristics and a fast simulator.
- Bayesian beta-binomial go/no-go interim with predictive probability of success, exact
  operating characteristics, calibration and assurance.
- `design`, `simulate`, `bayes` and `report` commands, TOML config files, Markdown report with
  plots, and a design memo.
