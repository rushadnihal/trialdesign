# trialdesign

Group-sequential and Bayesian design for two-arm clinical trials: fixed-design sample size,
Lan-DeMets stopping boundaries with non-binding futility, a beta-binomial go/no-go interim,
and Monte Carlo operating characteristics. Every number is validated against R (`rpact`, `gsDesign`).

Status: work in progress. Done: M0 skeleton, M1 fixed designs, M2 spending functions and
group-sequential boundaries.

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
    [1/3, 2/3, 1.0], alpha_one_sided=0.025, power=0.80,
    efficacy_spending="obf", futility_spending="obf",  # futility is non-binding
)
d.efficacy, d.futility, d.inflation_factor
d.max_sample_size(fixed.events_survival(0.70).events)  # max events = fixed events x IF
```

## Reference values

`tests/reference_values.json` is generated in R (needs `gsDesign` and `rpact`):

```bash
Rscript tests/r/reference_values.R > ref.json && mv ref.json tests/reference_values.json
```

## License

MIT
