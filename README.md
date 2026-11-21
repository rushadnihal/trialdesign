# trialdesign

Group-sequential and Bayesian design for two-arm clinical trials: fixed-design sample size,
Lan-DeMets stopping boundaries with non-binding futility, a beta-binomial go/no-go interim,
and Monte Carlo operating characteristics. Every number is validated against R (`rpact`, `gsDesign`).

Status: work in progress. Done: M0 skeleton, M1 fixed designs, M2 spending functions and
group-sequential boundaries, M3 simulated operating characteristics.

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

## Reference values

`tests/reference_values.json` is generated in R (needs `gsDesign` and `rpact`):

```bash
Rscript tests/r/reference_values.R > ref.json && mv ref.json tests/reference_values.json
```

## License

MIT
