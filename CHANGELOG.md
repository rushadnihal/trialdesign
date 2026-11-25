# Changelog

All notable changes to this project. Versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- **Patient-level survival simulator** (`trialdesign.patients.simulate_survival`): piecewise
  accrual (evenly spaced or random entry), permuted-block randomisation, exponential events and
  dropout, looks at the calendar time of the planned event counts, and the log-rank test.
  Reports power or type I error, mean look times, study duration, and expected patients and
  events, each with a Monte Carlo SE. Validated against `rpact::getSimulationSurvival`.
- **Patient-level normal and binary simulators** (`trialdesign.patients.simulate_normal`,
  `simulate_binary`), with non-inferiority margins (`delta0`). Normal: t-test with the pooled
  variance, turned into the z-value with the same p-value (`test="t"`, default) or compared
  with the boundaries as is (`test="z"`). Binary: z-test with the Farrington-Manning variance
  under H0. Validated against `rpact::getSimulationMeans` and `getSimulationRates`.
- `trialdesign.patients.simulate_design_patients`: H0 and H1 patient-level operating
  characteristics of a design, the counterpart of `simulate.simulate_design`.
- `trialdesign simulate --engine patient` (or `engine = "patient"` under `[simulation]`),
  default 10,000 trials. The `score` engine stays the default and its output is unchanged.
- A `slow` pytest marker for long simulation checks, run on demand with `pytest -m slow`.
- **Survival event re-estimation** (`trialdesign.ssr.survival_ssr`, `SurvivalSSR`): the
  promising-zone rule on the number of events, with the CHW test on log-rank score
  increments. Exact power, zones, expected events, study duration and patients; the extra
  events come from longer follow-up (`extend="followup"`, default) or longer accrual
  (`extend="accrual"`). Cap default 1.5 times the planned events. Validated against
  `rpact::getConditionalPower` and `rpact::getSimulationSurvival` with event re-estimation.
- `trialdesign.patients.simulate_survival_ssr`: patient-level simulation of that design,
  re-estimating the events at the interim log-rank z, with the CHW and the naive log-rank
  test. `patients.logrank_score` returns the log-rank score and its variance.
- `trialdesign ssr` and `trialdesign report` accept `--endpoint survival` with the accrual
  options, and `--extend` (or `extend` under `[adaptation]`); `--nsim` then runs the
  patient-level check.

### Changed

- `simulate.simulate_binary` moved to `patients.simulate_binary`; the old name still works
  and gives the same numbers.

## 0.2.0 (2026-11-23)

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
