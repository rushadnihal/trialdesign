# Design memo: TD-101 phase 2, Bayesian go/no-go interim

*Fictional trial, written to show how `trialdesign` supports a design decision. Sections 1 to 5
were written with `trialdesign` 0.1.0 and the addendum with 0.2.0; version 0.2.0 reproduces
every number, using the commands in the appendix.*

**To:** TD-101 clinical team · **From:** Trial statistics · **Decision needed:** sample size and
interim rule for the phase 2 study

## 1. The question

TD-101 is a new oral treatment being compared with standard of care. The primary endpoint is
clinical response at week 12, which is binary. About 30% of patients respond on standard of
care. The team considers a 15-point improvement, to 45%, clinically worthwhile and enough to
justify phase 3.

Phase 2 has to answer one question: **is TD-101 good enough to start phase 3?** It should say
yes when the true response rate is 45%, say no when TD-101 adds nothing, and stop early when the
answer is already clear.

## 2. The design

The design is two arms, 1:1, with one interim analysis at half enrolment and one final analysis.

- **Analysis model.** Beta(1, 1) (uniform) priors on each arm's response rate. The posterior is
  beta-binomial.
- **Final success.** The trial succeeds if P(p_T > p_C | data) ≥ η at the final analysis.
- **Interim.** Compute the predictive probability of success (PPoS): the chance the trial would
  succeed at the final analysis, averaged over the posterior predictive distribution of the
  patients still to come. **Go** (stop, recommend phase 3) if PPoS ≥ 0.9. **No-go** (stop for
  futility) if PPoS ≤ 0.1. Otherwise continue.
- **Calibration.** η is not chosen by hand. It is the smallest value whose *frequentist* type I
  error, P(success when p_T = p_C), stays at or below a target at every common response rate from
  0.05 to 0.95. Everything below is exact: the operating characteristics sum binomial
  probabilities over the decision tables, and a 200,000-trial simulation agrees within Monte
  Carlo error.

## 3. Operating characteristics

H0 means p_T = p_C = 0.30. H1 means p_T = 0.45. Assurance averages the probability of success
over a design prior p_T ~ Beta(4.5, 5.5), which has mean 0.45 and allows for uncertainty in the
effect (O'Hagan, Stevens and Campbell 2005). N counts patients in both arms.

| Design | A: package default | B | **C: recommended** | D |
| --- | --- | --- | --- | --- |
| Patients per arm (interim / final) | 30 / 60 | 30 / 60 | **50 / 100** | 90 / 180 |
| Type I error target (one-sided) | 0.025 | 0.10 | **0.10** | 0.025 |
| Success threshold η | 0.98 | 0.9068 | **0.9071** | 0.98 |
| Type I error at p_C = 0.30 (worst case) | 0.023 (0.023) | 0.100 (0.100) | **0.100 (0.100)** | 0.024 (0.025) |
| **Power at p_T = 0.45** | 0.36 | 0.63 | **0.79** | 0.81 |
| Assurance | 0.42 | 0.56 | **0.63** | 0.60 |
| P(go at interim), H1 | 0.11 | 0.25 | **0.37** | 0.39 |
| P(no-go at interim), H0 | 0.75 | 0.56 | **0.54** | 0.72 |
| P(no-go at interim), H1 | 0.29 | 0.14 | **0.07** | 0.06 |
| Expected N under H0 / H1 (maximum) | 74 / 96 (120) | 85 / 97 (120) | **142 / 156 (200)** | 230 / 280 (360) |

**Design A**, the package default, holds the type I error at one-sided 0.025. With 60 patients
per arm, however, it has only 36% power: it would miss a real 15-point improvement nearly two
times in three. To control 0.025 *and* reach 80% power, the trial needs about 180 per arm
(**design D**). That is close to the size of a phase 3 trial.

**Design C** treats phase 2 as what it is, a screen before a confirmatory trial. It accepts a
10% one-sided false-go rate, which is a common convention in phase 2 because phase 3 will
control 0.025 on its own. In return it gets 79% power with 100 patients per arm. Under H0 it
stops for futility at the interim more than half the time, so it uses 142 patients on average
instead of 200.

![Exact operating characteristics of design C against the true treatment response rate](img/memo_oc.png)

**The interim rule for design C**, in counts the team can read off directly, applies when
50 patients per arm have a week-12 response:

- **No-go** if the treatment arm has *no more* responders than control (for example 15 vs 15).
- **Go** if the treatment arm leads by about 10 responders (for example 25 or more vs 15).
- **Continue** to 100 per arm otherwise.

![Interim go / continue / no-go regions for design C](img/memo_decision_map.png)

## 4. Risks and how the design handles them

- **The control rate may not be 30%.** The type I error is calibrated at its worst case over
  every common response rate, so it holds whatever the control rate turns out to be. Power is
  stable for a 15-point gain: 0.82 at 25% vs 40%, and 0.78 at 35% vs 50%.
- **The effect may be smaller.** At 30% vs 40% power falls to 0.56, and at 30% vs 35% to 0.29.
  Design C is built to detect 15 points. A 5-point gain will usually, and rightly, lead to no-go.
- **No-go must be followed.** Unlike the non-binding futility of the group-sequential designs,
  the no-go rule is part of the type I error calculation. If the team overrides a no-go and
  continues, the type I error at p_C = 0.30 rises from 0.100 to 0.106. The DMC charter should make no-go
  binding, or η should be recalibrated with no-go ignored.
- **An early go overstates the effect.** Under H1 (true difference 15 points), trials that stop
  for go show an average observed difference of about 25 points at the interim. Phase 3 should
  be sized on the full phase 2 estimate shrunk toward the prior, or on 15 points, never on the
  interim estimate.
- **Response takes 12 weeks.** Patients keep enrolling while the interim cohort reaches week 12.
  Their data should be added at the final analysis (not ignored), and the interim should be
  scheduled when 50 per arm have a week-12 assessment, not when 50 per arm are enrolled.
- **Model scope.** Dropout, stratification and covariates are not modelled. With 10% dropout,
  enrol about 112 per arm to keep 100 evaluable.

## 5. Recommendation

Run **design C**: 100 evaluable patients per arm with an interim at 50 per arm. Use go if
PPoS ≥ 0.9, no-go if PPoS ≤ 0.1, and final success if P(p_T > p_C | data) ≥ 0.9071, a threshold
calibrated to a worst-case one-sided false-go rate of 10%. Make the no-go rule binding in the DMC
charter. If governance requires phase 2 to control one-sided 0.025, the honest alternative is
design D (180 per arm), not the 60-per-arm default, which is underpowered for this effect.

## Addendum (v0.2): protecting phase 3 against a smaller effect

Section 4 warns that an early go overstates the effect. Phase 3 can be sized on the
15-point improvement and still guard against a smaller true effect, by re-estimating its
sample size at the interim (Mehta and Pocock 2011).

**Planned phase 3.** 30% vs 45%, one-sided alpha 0.025, 80% power, an interim at half the
patients with O'Brien-Fleming-type efficacy and non-binding futility: 344 patients (172 per
arm), interim at 172. **Rule.** At the interim, compute the conditional power at the observed
trend. If it is between 0.36 and 0.80 (the promising zone), enrol enough extra patients to
bring it to 0.80, up to twice the plan (688). Otherwise keep 344. The final analysis weights
the two stages by their planned sizes (Cui, Hung and Wang 1999), so the type I error stays
exactly 0.025 (0.0249 in 200,000 simulated trials).

| True improvement | Power, planned | Power, with re-estimation | Mean patients, planned | Mean patients, with re-estimation |
| --- | --- | --- | --- | --- |
| 15 points (as designed) | 0.80 | 0.83 | 302 | 336 |
| about 13 points | 0.66 | 0.71 | 304 | 344 |
| about 10.5 points | 0.50 | 0.55 | 300 | 342 |

The smaller improvements are 85% and 70% of the design effect on the z-scale. For example, an
interim z of 1.5 falls in the promising zone: conditional power at the observed trend is 0.59,
and the rule raises the trial to 518 patients.

**What it buys.** About 4 to 5 points of power for about 40 more patients on average. That is
cheap insurance if the team believes 15 points but fears 12. If 12 points is the effect phase 3
must detect, size for it from the start: 528 patients at most, 463 on average under that
effect, with no adaptive machinery to explain to regulators.

## Appendix: reproduce every number

```bash
# Design A (package default, calibrated to one-sided 0.025)
trialdesign bayes --pc 0.30 --pt 0.45 --n 60 --interim 30 --design-prior-t 4.5,5.5
# Design B and C (calibrated to one-sided 0.10)
trialdesign bayes --pc 0.30 --pt 0.45 --n 60 --interim 30 --calibrate --alpha-one-sided 0.10 \
    --design-prior-t 4.5,5.5
trialdesign bayes --pc 0.30 --pt 0.45 --n 100 --interim 50 --calibrate --alpha-one-sided 0.10 \
    --design-prior-t 4.5,5.5 --nsim 200000 --seed 2026
# Design D
trialdesign bayes --pc 0.30 --pt 0.45 --n 180 --interim 90 --design-prior-t 4.5,5.5
# Full report with plots for design C
trialdesign report --pc 0.30 --pt 0.45 --n 100 --interim 50 --calibrate --alpha-one-sided 0.10 \
    --design-prior-t 4.5,5.5 --out results/memo
# Addendum: phase 3 with sample size re-estimation, the rule at z1 = 1.5, and a 12-point design
trialdesign ssr --endpoint binary --pc 0.30 --pt 0.45 --futility obf --nsim 200000 --seed 2026
trialdesign ssr --endpoint binary --pc 0.30 --pt 0.45 --futility obf --z1 1.5
trialdesign design --endpoint binary --pc 0.30 --pt 0.42 --looks 0.5,1 --futility obf
```

**References.** Berry, Carlin, Lee and Müller (2010), *Bayesian Adaptive Methods for Clinical
Trials*. Cui, Hung and Wang (1999), *Biometrics* 55, 853–857. Lee and Liu (2008), *Clinical
Trials* 5, 93–106. Mehta and Pocock (2011), *Statistics in Medicine* 30, 3267–3284. O'Hagan,
Stevens and Campbell (2005), *Pharmaceutical Statistics* 4, 187–201. Spiegelhalter, Abrams and
Myles (2004), *Bayesian Approaches to Clinical Trials and Health-Care Evaluation*.
