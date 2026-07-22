# Minutes Extension ML System (v2)

Predicts whether an NBA player is likely to play significantly more minutes than usual in an upcoming game — and, just as importantly, when the model doesn't know enough to say.

This is a rebuild of an earlier prototype. The first version used synthetic in-game data and assumed a coach as the end user. This version uses real NBA game logs, real held-out evaluation, and a different target user — and getting to that reframe is as much the point of this project as the model itself.

---

## Problem Framing

The original version of this project modeled the problem as "would a coach want this?" That framing doesn't hold up: a coach already knows in real time when a teammate is injured, in foul trouble, or when the game is a blowout. A model telling a coach "extended minutes are likely" adds nothing they don't already know from being in the building.

The people who *don't* have that inside information but still need to predict minutes before tip-off are daily fantasy players, prop bettors, and fantasy analysts. They work from the same public data this model uses — box scores, rest days, home/away, recent workload — and even when they know a teammate is out (from the public injury report), it's often not obvious which bench player absorbs the extra minutes, or how many. That's a real, non-trivial prediction problem for this audience in a way it isn't for a coach.

This matters beyond framing: it also changes how prediction errors should be weighed. A false "yes" here means someone acts on bad information — rosters a player or takes a prop bet expecting a big minutes jump that doesn't happen. A false "no" just means a missed opportunity. That asymmetry (a wrong "yes" costs real money; a missed "yes" doesn't) is why this project uses an abstention layer instead of forcing a yes/no call on every game — see below.

---

## Data

- Source: `nba_api` (`playergamelog`), pulled directly from NBA.com stats endpoints.
- Players: 9 rotation/bench players spanning multiple roles — Alex Caruso, Payton Pritchard, Naz Reid, Davion Mitchell, Rui Hachimura, Austin Reaves, Anfernee Simons, Obi Toppin, Gary Trent Jr.
- Seasons: 2023, 2024, 2025 regular seasons.
- Result: 1,573 total player-games after feature engineering.
- Split: held out by player, not by row. Three players (IDs `1630558`, `1630202`, `1629675`) are entirely excluded from training and used only for testing, so the model is evaluated on players it has never seen rather than on unseen games from players it already knows. 968 training rows, 605 test rows.

## Label

A game counts as "extended minutes" if the player played at least 10% more than their trailing 10-game rolling average:

```
label = 1 if MIN > rolling_avg_min * 1.10 else 0
```

Class balance is roughly 66% no / 34% yes in both splits.

## Features

All features are computed only from information available *before* the game being predicted — no same-game stats are used, which would leak the outcome.

| Feature | Description |
|---|---|
| `prev_min` | Minutes played in the immediately prior game |
| `days_rest` | Days between the prior game and this one |
| `is_away` | Whether this game is on the road |
| `rolling_avg_min` | Trailing 10-game average minutes, computed with a 1-game shift so it never includes the game being predicted |

Because `rolling_avg_min` needs 10 prior games to be valid, each player's first 10 games of available history are dropped rather than computed on a partial window.

---

## Model & Evaluation

Logistic regression (scikit-learn), features standardized with `StandardScaler` fit on train only.

Real held-out test set results (605 games, 3 unseen players):

```
              precision    recall  f1-score   support
           0       0.69      0.93      0.79       401
           1       0.55      0.18      0.27       204

    accuracy                           0.67       605
```

Confusion matrix:

```
[[371  30]
 [167  37]]
```

**Honest read of this result:** the model is conservative — it rarely predicts "yes," and when it does, it's right about as often as a coin flip (55% precision). It catches only 18% of real extended-minutes games. Checked directly: across the entire test set, only 2 predicted probabilities ever exceed 0.7, and the single highest probability the model ever produces is 0.76. This isn't a threshold-tuning problem — the model is structurally not confident about class 1 with these four features. Reweighting the loss (`class_weight='balanced'`) would shift where the decision boundary falls, but can't manufacture separation that isn't present in the features. The real next lever is better features (see Limitations & Next Steps), not threshold or weighting tricks.

---

## Abstention Policy

Given the model's actual confidence ceiling, forcing a yes/no call on every game at the default 0.5 cutoff isn't honest — most of those calls are near coin-flip odds. `policy.py` adds a third state instead of two:

- probability `>= high` → commit to **yes**
- probability `<= low` → commit to **no**
- otherwise → **abstain** — not enough signal, no call

This is deliberately written in plain functions (no classes, minimal dependencies) so the logic is easy to audit line by line — coverage, precision, and recall on only the committed cases are computed directly with loops and counters, not a metrics library.

Real calibration sweep on the held-out test set:

```
  low  high  coverage  games  precision  recall
--------------------------------------------------
  0.5   0.5       1.0    605      0.552   0.181
  0.4   0.6     0.734    444      0.583   0.105
 0.35  0.65      0.61    369      0.600   0.059
  0.3   0.7     0.428    259      1.000   0.037
 0.25  0.75      0.24    145      1.000   0.036
```

Widening the band raises precision on committed calls, but recall collapses even faster than precision improves, and the perfect-looking 1.0 precision at wider bands comes from only a handful of "yes" calls (the model produces almost none above 0.7 in the first place). The honest takeaway is the abstention layer correctly protects against acting on weak signal, but it cannot rescue a model that has this little separating power for the positive class to begin with.

---

## Repository Structure

```
build_features.py   Pulls game logs via nba_api, computes leakage-safe features and label, writes features.csv
explore.py           Scratch script used to inspect raw nba_api output shape/columns
features.csv         1,573-row engineered dataset (9 players, 3 seasons)
train.py             Loads features.csv, splits by held-out players, trains logistic regression, evaluates
policy.py             Abstention policy layer (decide / coverage / calibration sweep) applied to model probabilities
```

**Not yet migrated:** the FastAPI service, Docker container, and JSONL decision logging built for the v1 prototype still reflect the old synthetic feature set (`fatigue_index`, `fouls`, `time_left`, etc.) and have not been updated to serve this model. That migration is the next planned step — see below.

---

## Engineering Decisions

- **Held out by player, not by row:** a random row-level split would let the model see other games from the same player it's being tested on, which overstates real-world performance. Holding out entire players is a harder, more honest test of generalization.
- **Logistic regression over a more complex model:** chosen for interpretable, well-calibrated probabilities, which the abstention layer depends on directly. A more complex model might raise accuracy but would make the probability outputs harder to reason about and trust for a threshold-based decision system.
- **Abstention over a forced threshold:** given the cost asymmetry for this audience (a wrong "yes" costs money, a missed "yes" doesn't), silently forcing a call at 0.5 would misrepresent confidence the model doesn't have. Abstaining is the more honest design even though it reduces how often the model is useful.
- **Plain-function implementation of the policy layer:** written without classes or a metrics library so every precision/recall number in the calibration table can be traced back to a simple loop, not a black-box call.

## Limitations

- Only 9 players and 3 seasons — a small, non-representative sample. Results likely don't generalize to the full league without a much larger player pool.
- The four current features (`prev_min`, `days_rest`, `is_away`, `rolling_avg_min`) capture recent workload and schedule, but nothing about *why* a role would expand — teammate injury/inactive status, foul trouble, or blowout margin aren't in the model yet, despite being the most likely real drivers of extended minutes.
- Recall on the positive class is weak (0.18 at 0.5, dropping further with abstention) and the model rarely produces a probability above 0.7 at all — this is a signal/feature ceiling, not a tuning problem.
- The serving layer (API, Docker, logging) has not yet been updated to reflect this model or feature set.

## Next Steps

- Add public injury/inactive status for the player's team as a feature — likely the single highest-leverage addition, and unlike other candidate features, it's information the target audience (bettors/DFS players) actually has access to pre-game.
- Reconsider the framing of the label itself: rather than a per-player binary "extended or not," modeling *which* of several eligible bench candidates absorbs open minutes, and by how much, may better match what this audience actually needs to act on.
- Migrate the FastAPI/Docker/logging layer to serve this model and feature set, replacing the v1 synthetic-data version.
- Expand the player/season sample before treating any result here as representative.

## Running the Code

```bash
pip install -r requirements.txt   # nba_api, pandas, scikit-learn, numpy

python build_features.py          # pulls data via nba_api, writes features.csv
python train.py                   # trains model, prints evaluation + abstention calibration table
```
