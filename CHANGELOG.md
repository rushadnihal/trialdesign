# Changelog

All notable changes to this project. Versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- **Non-proportional hazards and Weibull event times in the patient-level survival simulator**
  (`trialdesign.hazards`): `EventModel.delayed_effect`, `waning_effect`, `piecewise` (any
  piecewise-constant hazards per arm) and `weibull` (proportional hazards with a Weibull
  baseline, `rpact`'s parameterisation). `patients.simulate_survival(..., model=...)` and
  `simulate_design_patients(..., model=...)` draw event times by inverting the cumulative
  hazard; H0 uses `model.null()`. One hazard interval gives exactly the v0.3 trials for the
  same seed. Validated against `rpact::getSimulationSurvival` with `piecewiseSurvivalTime` and
  `kappa`.
- `hazards.expected_events` and a `model` argument to `SurvivalStudy.events_at`,
  `time_of_events` and `look_times`: expected events and look times under any of those
  models, validated against `rpact::getEventProbabilities`.
- **Fleming-Harrington weighted log-rank test** (`patients.weighted_logrank_z`, and
  `simulate_survival(..., fh=(rho, gamma))`) for a single final analysis. Validated against
  an R implementation and `survival::survdiff`.
- `trialdesign simulate --engine patient` takes `--delay`, `--waning`, `--hr-after`,
  `--hazard-times`, `--hazards-control`, `--hazards-treatment`, `--kappa` (or a `[hazards]`
  table) and `--test fh:<rho>,<gamma>` (or `test` under `[simulation]`). JSON output names the
  event model and the test.
- `examples/phase3_survival_nph.toml`: a design sized for HR 0.70 and simulated with a
  4-month delayed effect.
- **Graphical multiple testing** (`trialdesign.multiplicity`, Bretz et al. 2009): `Graph`
  (weights and transition matrix, with Bonferroni, Holm, fixed-sequence, fallback and
  two-dose presets), `graph_test` (rejections, adjusted p-values and the path through the
  graph) and `graph_power` (simulated power per hypothesis, of at least one and of all, from
  correlated z-statistics). Validated against a closed-test implementation in R,
  `stats::p.adjust`, and exact power from `mvtnorm::pmvnorm`.
- `trialdesign graph` command and a `[graph]` table in design files;
  `examples/two_doses_graph.toml`.

## 0.3.0 (2026-11-26)

A patient-level simulator next to the score-process engine, survival sample size re-estimation
on the events scale, end-of-trial inference and equivalence designs. v0.2 config files and
Python calls keep working, and the default (score) engine gives the same numbers as before.

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
- **End-of-trial inference** (`trialdesign.inference.analyse`): repeated confidence intervals
  and p-values at every look, and, once a boundary is crossed or at the last look, the
  stage-wise ordering p-value, median-unbiased estimate and confidence interval. Binding
  futility is followed, non-binding ignored. Validated against `rpact`
  (getRepeatedConfidenceIntervals, getRepeatedPValues, getFinalPValue,
  getFinalConfidenceInterval). From the command line: `trialdesign design --looks ...
  --analyse z1,z2,...` (or `analyse` under `[sequential]`).
- **Equivalence by two one-sided tests** for fixed normal and binary designs
  (`trialdesign.equivalence`): sample size, power and the test itself, with symmetric or
  asymmetric margins. `trialdesign design --margin` (or `margin` under `[design]`).
- Python 3.13 in the CI matrix and the package classifiers.
- `trialdesign report --engine patient` puts a 10,000-trial patient-level simulation next to
  the score-engine one and the exact values; `report --analyse` adds an inference section.
- `examples/phase3_survival_ssr.toml`, and a v0.3 addendum to the design memo: a patient-level
  check of the phase 3 binary design (its z-test is slightly liberal at 172 patients) and how
  to report the end of a group-sequential trial.

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
