# trialdesign

Group-sequential and Bayesian design for two-arm clinical trials: fixed-design sample size,
Lan-DeMets stopping boundaries with non-binding futility, a beta-binomial go/no-go interim,
and Monte Carlo operating characteristics. Every number is validated against R (`rpact`, `gsDesign`).

Status: work in progress (milestone M0, project skeleton).

## Install (development)

```bash
pip install -e ".[dev]"
trialdesign --version
pytest
ruff check . && ruff format --check .
```

## License

MIT
