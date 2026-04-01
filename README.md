# 🏆 Wave Your Flag! — Predicting FIFA World Cup 2026

A Monte Carlo simulation of the 2026 FIFA World Cup, built on top of a machine learning pipeline trained on international football results going back to 1872.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mohamedfarouk94/wave-your-flag-predicting-fifa-world-cup-2026)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

The simulator runs the entire 2026 World Cup — all 12 groups, the Round of 32, Round of 16, quarter-finals, semi-finals, and final — tens of thousands of times. Each run uses a trained ML model to estimate expected goals, a statistically grounded score generator to produce actual scorelines, and an Elo-style rating system that updates team strengths throughout the tournament. The aggregated results yield probabilities: how likely is each team to win, reach the final, or exit in the group stage?

---

## How It Works

The pipeline has five stages:

1. **Historical ratings** — Every team is assigned an Elo-style rating, an Attack Strength Indicator (ASI), and a Defense Weakness Indicator (DWI), computed iteratively from over 150 years of international match results.

2. **xG prediction** — A `XGBRegressor` with a Poisson objective predicts expected goals (λ) for each team in a match, using six features: team rating, opponent rating, team ASI, opponent DWI, match importance, and venue weight.

3. **Score generation** — A custom **1X2-Corrected Poisson Generator** samples scorelines from the predicted λ values. It first draws the match outcome (home win / draw / away win) using Skellam-distribution probabilities — with a tunable shock factor `μ` that compresses upset probability — then samples a Poisson-consistent score for that outcome.

4. **Tournament simulation** — The full competition is modelled with a clean OOP architecture (`Team`, `Side`, `Match`, `Group`, `GroupStage`, `KnockoutRound`, `Tournament`). A parameter `ρ ∈ [0, 1]` controls whether ratings update dynamically after each simulated match.

5. **Monte Carlo aggregation** — The `SimulationManager` runs ~100,000 independent tournament instances and aggregates championship probabilities, round-by-round exit rates, and match statistics.

For a detailed walkthrough, see the [Kaggle notebook](https://www.kaggle.com/code/mohamedfarouk94/wave-your-flag-predicting-fifa-world-cup-2026).

---

## Project Structure

```
fwc26/
│
├── data/
│   ├── teams_data.csv              # Pre-computed ratings, ASI, DWI for all 48 WC26 teams
│   ├── best_thirds_fwc26.csv       # Third-place qualifier bracket combinations
│   └── flags/                      # Flag images (PNG) indexed by ISO alpha-2 code
│
├── models/
│   ├── model.json                  # Trained Poisson XGBRegressor
│   └── params.joblib               # Model hyperparameters
│
├── simulation.py                   # Core simulation engine (all classes + WC26 builder)
|── backend.py                      # FastAPI backend
|── cli                             # Simple CLI
|── index.html                      # Frontend home pge
|── simulate.html                   # Frontend simulation page
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/fwc26.git
cd fwc26
pip install -r requirements.txt
```

**Requirements:**

```
numpy
pandas
scipy
scikit-learn
xgboost
joblib
matplotlib
plottable
tqdm
```

---

## Usage

### Run a single simulation

```python
from simulation import run_sim

run_sim(rho=0.5, mu=0.5)
```

`rho` controls how much team ratings update during the tournament (`0` = fully static, `1` = fully dynamic). `mu` is the upset dampening factor — lower values make upsets less likely.

### Run the full Monte Carlo (programmatic)

```python
from simulation import SM, wc26_builder

sm = SM(n=10_000, year=2026, wc_builder=wc26_builder, rho=0.5, mu=0.5)
sm.run(verbose=1)

# Championship probabilities
report = sm.report_
for team, prob in sorted(report['champion_record'].items(), key=lambda x: -x[1]):
    print(f"{team}: {prob:.1%}")
```

### Access a single tournament run

```python
sm = SM(n=1, year=2026, wc_builder=wc26_builder)
sm.run()

wc = sm.trs_[0].wc
wc.plot()                       # Print group tables and knockout scoreboard
wc.Champion.get_team()          # The champion of this run
wc.F.M1.get_score()             # Final scoreline
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | `0.5` | Dynamic rating update strength. `0` = frozen ratings throughout the tournament. `1` = full update after every match. |
| `mu` | `0.5` | Upset probability dampener. `0` = favourites almost always win. `1` = raw Poisson (no correction). |
| `n` | `1000` | Number of Monte Carlo simulation runs. `~100,000` recommended for stable probabilities. |

---

## Backtesting

The model was validated against the 2014, 2018, and 2022 World Cups by running each edition with pre-tournament ratings and comparing simulated distributions to actual results. Evaluation metrics include:

- **RBMSE** (Ranking-Based Mean Squared Error) — how well simulated team progressions match expectation given pre-tournament strength.
- Goals per match, draw ratio, 0-0 ratio, heavy-score ratio, and goal party ratio — compared to historical distributions.

The 2022 World Cup (Argentina winning) was the most "expected" edition of the last three by RBMSE. 2014 and 2018 were significantly more chaotic.

---

## Acknowledgements

- **Dataset**: [International Football Results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) by Mart Jürisoo
- **Flags**: [flagsapi.com](https://flagsapi.com/) + Wikimedia Commons for special cases
- **Rating system**: Inspired by [FIFA's official Men's World Ranking methodology](https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking#Calculation_method)
- **Goal generation**: Builds on ideas from the [Maher (1982)](https://en.wikipedia.org/wiki/Statistical_association_football_predictions#Time-Independent_Poisson_Regression) and [Dixon–Coles (1997)](https://academic.oup.com/jrsssc/article/46/2/265/6990584) football modelling literature

---

## License

MIT License. See [LICENSE](LICENSE) for details.