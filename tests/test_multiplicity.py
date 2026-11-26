"""M18: graphical multiple testing (Bretz et al. 2009).

Reference values in ``tests/multiplicity_reference_values.json`` come from
``tests/r/multiplicity_reference_values.R``: an independent R implementation in the
closed-test form (every intersection hypothesis tested by its weighted Bonferroni test),
checked there against ``stats::p.adjust(method = "holm")`` for Holm graphs, and exact power by
integrating the multivariate normal (``mvtnorm::pmvnorm``) over the cells cut by the critical
values. Simulated power to reject at least one hypothesis must sit within 2 Monte Carlo SEs
of the exact value; the per-hypothesis and all-rejected rates within 3, because there are
about 25 of them from the same draws (at 2 SEs a few would fail by chance alone), as in
``test_patients.py``.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from typer.testing import CliRunner

from trialdesign.cli import app
from trialdesign.multiplicity import Graph, _reject_batch, _remove, graph_power, graph_test

REF = json.loads((Path(__file__).parent / "multiplicity_reference_values.json").read_text())
EXAMPLE = str(Path(__file__).parents[1] / "examples" / "two_doses_graph.toml")


def _graph(case: dict) -> Graph:
    return Graph(tuple(case["weights"]), tuple(tuple(r) for r in case["transitions"]))


# ----------------------------------------------------------------------------- vs R


@pytest.mark.parametrize("case", REF["tests"], ids=[c["id"] for c in REF["tests"]])
def test_rejections_and_adjusted_p_match_r(case):
    res = graph_test(case["p"], _graph(case), case["alpha_one_sided"])
    np.testing.assert_array_equal(res.rejected, case["rejected"])
    np.testing.assert_allclose(res.adjusted_p, case["adjusted_p"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", REF["power"], ids=[c["id"] for c in REF["power"]])
def test_power_matches_exact_integration(case):
    pw = graph_power(
        _graph(case), case["means"], case["correlation"], case["alpha_one_sided"], 200_000, 2026
    )
    for ours, se, exact in zip(pw.local_power, pw.local_power_se, case["local_power"], strict=True):
        assert abs(ours - exact) <= 3 * max(se, 1e-4), (ours, exact)
    assert abs(pw.any_power - case["any_power"]) <= 2 * pw.any_power_se
    assert abs(pw.all_power - case["all_power"]) <= 3 * max(pw.all_power_se, 1e-4)
    assert pw.expected_rejections == pytest.approx(case["expected_rejections"], abs=0.01)


def test_global_null_fwer_is_below_alpha():
    case = next(c for c in REF["power"] if c["id"] == "two_doses_global_null")
    assert case["any_power"] <= 0.025


# ------------------------------------------------------------------------- properties


def _random_graph(rng: np.random.Generator, m: int) -> Graph:
    w = rng.dirichlet(np.ones(m)) * rng.uniform(0.8, 1.0)
    g = rng.dirichlet(np.ones(m - 1), size=m) * rng.uniform(0.5, 1.0, size=(m, 1))
    full = np.zeros((m, m))
    for i in range(m):
        full[i, [j for j in range(m) if j != i]] = g[i]
    return Graph(tuple(w), tuple(tuple(r) for r in full))


def test_result_does_not_depend_on_rejection_order():
    """Rejecting the rejectable hypotheses in any order ends in the same set."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        m = int(rng.integers(2, 6))
        graph = _random_graph(rng, m)
        p = rng.uniform(0, 0.03, m)
        expected = graph_test(p, graph).rejected
        for order in itertools.permutations(range(m)):
            w = np.array(graph.weights)[None, :]
            g = np.array(graph.transitions)[None, :, :]
            done = np.zeros(m, dtype=bool)
            changed = True
            while changed:
                changed = False
                for j in order:
                    if not done[j] and w[0, j] > 0 and p[j] <= w[0, j] * 0.025:
                        w, g = _remove(w, g, np.array([j]))
                        done[j] = changed = True
                        break  # rescan in the same fixed order
            np.testing.assert_array_equal(done, expected)


def test_holm_matches_textbook_holm():
    rng = np.random.default_rng(2)
    for m in (2, 3, 5):
        for _ in range(20):
            p = rng.uniform(0, 0.06, m)
            order = np.argsort(p)
            textbook = np.maximum.accumulate(np.minimum(1, p[order] * (m - np.arange(m))))
            adj = np.empty(m)
            adj[order] = textbook
            np.testing.assert_allclose(graph_test(p, Graph.holm(m)).adjusted_p, adj, atol=1e-14)


def test_fixed_sequence_stops_at_the_first_failure():
    res = graph_test([0.01, 0.03, 0.001], Graph.fixed_sequence(3))
    np.testing.assert_array_equal(res.rejected, [True, False, False])
    np.testing.assert_allclose(res.adjusted_p, [0.01, 0.03, 0.03])


def test_bonferroni_passes_nothing_on():
    res = graph_test([0.008, 0.009, 0.5], Graph.bonferroni(3))
    np.testing.assert_array_equal(res.rejected, [True, False, False])
    np.testing.assert_allclose(res.adjusted_p, [0.024, 0.027, 1.0])


def test_intersection_weights_of_two_doses():
    g = Graph.two_doses()
    np.testing.assert_allclose(g.weights_for([1, 2, 3]), [0, 0.75, 0.25, 0])
    np.testing.assert_allclose(g.weights_for([2, 3]), [0, 0, 0.5, 0.5])
    np.testing.assert_allclose(g.weights_for([0, 1, 2, 3]), g.weights)


def test_steps_record_the_path():
    res = graph_test([0.004, 0.015, 0.011, 0.2], Graph.two_doses())
    assert [s["rejected"] for s in res.steps] == [
        "H1 primary high",
        "H2 primary low",
        "H3 secondary high",
    ]
    assert res.steps[0]["level"] == pytest.approx(0.0125)
    np.testing.assert_allclose(res.steps[-1]["weights_after"], [0, 0, 0, 1])


def test_simulated_fwer_under_partial_nulls():
    """Strong control: with some hypotheses true, false rejections stay below alpha."""
    g = Graph.two_doses()
    corr = 0.5
    for means in ([3.0, 0, 0, 0], [3.0, 3.0, 0, 0], [0, 0, 3.0, 3.0], [3.0, 0, 3.0, 0]):
        pw = graph_power(g, means, corr, n_sim=100_000, seed=3)
        true_null = np.array(means) == 0
        rng = np.random.default_rng(4)
        r = np.full((4, 4), corr)
        np.fill_diagonal(r, 1)
        z = np.array(means) + rng.multivariate_normal(np.zeros(4), r, 100_000)
        rej = _reject_batch(stats.norm.sf(z), g, 0.025)
        fwer = rej[:, true_null].any(axis=1).mean()
        assert fwer <= 0.025 + 2 * np.sqrt(0.025 * 0.975 / 100_000), (means, fwer)
        assert pw.n_sim == 100_000


def test_input_checks():
    with pytest.raises(ValueError, match="sum to at most 1"):
        Graph((0.7, 0.7), ((0, 1), (1, 0)))
    with pytest.raises(ValueError, match="zero diagonal"):
        Graph((0.5, 0.5), ((0.5, 0.5), (1, 0)))
    with pytest.raises(ValueError, match="sum to at most 1"):
        Graph((0.5, 0.5, 0), ((0, 0.6, 0.6), (1, 0, 0), (1, 0, 0)))
    with pytest.raises(ValueError, match="2 x 2"):
        Graph((0.5, 0.5), ((0, 1, 0), (1, 0, 0)))
    with pytest.raises(ValueError, match="2 p-values"):
        graph_test([0.01], Graph.holm(2))
    with pytest.raises(ValueError, match="positive semi-definite"):
        graph_power(Graph.holm(3), [1, 1, 1], [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])


def test_seed_reproduces_power():
    a = graph_power(Graph.holm(3), [2.5, 2, 1], 0.3, n_sim=5_000, seed=9)
    b = graph_power(Graph.holm(3), [2.5, 2, 1], 0.3, n_sim=5_000, seed=9)
    np.testing.assert_array_equal(a.local_power, b.local_power)
    assert a.seed == 9


# ------------------------------------------------------------------------ command line

runner = CliRunner()


def test_cli_example_config_text():
    result = runner.invoke(app, ["graph", "--config", EXAMPLE, "--nsim", "20000"])
    assert result.exit_code == 0, result.output
    assert "1. reject H1 primary high (p 0.004 <= 0.0125)" in result.output
    assert "Reject at least one" in result.output


def test_cli_preset_json():
    result = runner.invoke(app, ["graph", "--preset", "holm:3", "--p", "0.01,0.04,0.02", "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["test"]["adjusted_p"] == pytest.approx([0.03, 0.04, 0.04])
    assert out["test"]["rejected"] == [False, False, False]
    assert out["power"] is None


def test_cli_matrix_flags_and_names():
    result = runner.invoke(
        app,
        [
            *("graph", "--weights", "0.5,0.5", "--transitions", "0,1;1,0"),
            *("--names", "a,b", "--means", "2.8,2.8", "--correlation", "0.5"),
            *("--nsim", "2000", "--seed", "1", "--json"),
        ],
    )
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert [h["name"] for h in out["hypotheses"]] == ["a", "b"]
    assert out["power"]["seed"] == 1


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--preset", "holm:3"), "--p to test"),
        (("--weights", "0.5,0.5", "--p", "0.1,0.1"), "--transitions"),
        (("--preset", "holm", "--p", "0.1,0.1"), "number of hypotheses"),
        (("--preset", "tree:3", "--p", "0.1"), "preset must be one of"),
        (("--preset", "holm:2", "--weights", "1,0", "--p", "0.1,0.1"), "not both"),
    ],
)
def test_cli_rejects_bad_input(args, message):
    result = runner.invoke(app, ["graph", *args])
    assert result.exit_code != 0
    assert message in result.output
