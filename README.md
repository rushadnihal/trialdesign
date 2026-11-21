# trialdesign

Group-sequential and Bayesian design for two-arm clinical trials: fixed-design sample size,
Lan-DeMets stopping boundaries with non-binding futility, a beta-binomial go/no-go interim,
and Monte Carlo operating characteristics. Every number is validated against R (`rpact`, `gsDesign`).

Status: work in progress. Done: M0 skeleton, M1 fixed designs, M2 spending functions and
group-sequential boundaries, M3 simulated operating characteristics, M4 Bayesian interim.

## Install (development)

```bash
pip install -e ".[dev]"
trialdesign --version
pytest
ruff check . && ruff format --check .
```

## Group-sequential boundaries (Python API)

```python
from trialdesign import boundaries, fixed

d = boundaries.group_sequential_design(
    [1 / 3, 2 / 3, 1.0],
    alpha_one_sided=0.025,
    power=0.80,
    efficacy_spending="obf",
    futility_spending="obf",  # futility is non-binding
)
d.efficacy, d.futility, d.inflation_factor
d.max_sample_size(fixed.events_survival(0.70).events)  # max events = fixed events x IF
```

## Operating characteristics by simulation (Python API)

```python
from trialdesign import simulate

sim = simulate.simulate_design(d, n_sim=200_000, seed=2026)
sim.type_i_error, sim.h0_ignore_futility.reject_prob_se  # futility ignored (non-binding)
sim.power, sim.h1.reject_prob_se
sim.h1.prob_efficacy, sim.h1.prob_futility  # per-look stopping probabilities
sim.h0.expected_sample_size(d.max_sample_size(fixed.events_survival(0.70).events))
```

The simulator draws independent normal increments of the score process, one
`(n_sim, looks)` matrix per batch, so 100,000 trials take well under a second. Type I error is
reported with futility ignored, which is the number that matters for a non-binding design;
`sim.h0` shows the same trials with futility followed.

## Bayesian go/no-go interim (Python API)

```python
from trialdesign import bayes

bayes.prob_superiority(6, 30, 12, 30)  # P(p_T > p_C | 6/30 vs 12/30), Beta(1, 1) priors

b = bayes.bayesian_interim_design(
    n_interim=30, n_final=60,          # control patients; treatment = allocation x control
    success_threshold=0.95,            # final success: P(p_T > p_C | data) >= 0.95
    go_threshold=0.9, nogo_threshold=0.1,
)
b.ppos(6, 12), b.decide(6, 12)         # predictive probability of success, go/no-go/continue
oc = b.operating_characteristics(0.30, 0.30)
oc.prob_success                        # exact type I error of the whole procedure
b.assurance(0.30, (4.5, 5.5)).prob_success  # Beta(4.5, 5.5) design prior on p_T
bayes.simulate_bayesian(b, 0.30, 0.45, n_sim=200_000, seed=2026).reject_prob
```

PPoS is computed exactly by enumerating the beta-binomial posterior predictive of the remaining
patients, so decisions carry no Monte Carlo error. Every decision is a lookup in two tables
(final success and interim decision), which gives exact operating characteristics and exact
assurance; the simulations, run through the M3 engine, check them.

## Reference values

`tests/reference_values.json` is generated in R (needs `gsDesign` and `rpact`):

```bash
Rscript tests/r/reference_values.R > ref.json && mv ref.json tests/reference_values.json
```

The M4 values in `tests/bayes_reference_values.json` need base R only:

```bash
Rscript tests/r/bayes_reference_values.R > bayes.json && mv bayes.json tests/bayes_reference_values.json
```

## License

MIT
