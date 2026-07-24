# Minutes Extension ML System (v2)


Predicts whether an NBA player is likely to play significantly more minutes than usual in an upcoming game — and, just as importantly, flags when the model doesn't have enough signal to say.

This is a rebuild of an earlier prototype. The first version used synthetic in-game data and assumed a coach as the end user. This version uses real NBA game logs, real held-out evaluation, and a different target user — and getting to that reframe is as much the point of this project as the model itself.

## Problem Framing

The original version modeled the problem as "would a coach want this?" That framing doesn't hold up: a coach already knows in real time when a teammate is injured, in foul trouble, or when a game is a blowout. A model telling a coach "extended minutes are likely" adds nothing they don't already know from being in the building.

The people who *don't* have that inside information but still need to predict minutes before tip-off are daily-fantasy players, prop bettors, and fantasy analysts. They work from the same public data this model uses — box scores, rest days, home/away, recent workload — and even when they know a teammate is out (from the public injury report), it's often not obvious which bench player absorbs the extra minutes, or how many. That's a real, non-trivial prediction problem for this audience in a way it isn't for a coach.

This also changes how prediction errors should be weighed. A false "yes" means someone acts on bad information — rosters a player or takes a prop bet expecting a minutes jump that doesn't come. A false "no" is just a missed opportunity. That asymmetry is why this project uses an abstention layer instead of forcing a yes/no call on every game (see below).

## Data

- **Source:** `nba_api` (`playergamelog`), pulled directly from NBA.com stats endpoints.
- **Players:** 9 rotation/bench players spanning multiple roles — Alex Caruso, Payton Pritchard, Naz Reid, Davion Mitchell, Rui Hachimura, Austin Reaves, Anfernee Simons, Obi Toppin, Gary Trent Jr.
- **Seasons:** 2023, 2024, 2025 regular seasons.
- **Result:** 1,573 total player-games after feature engineering.
- **Split:** held out by *player*, not by row. Three players (IDs 1630558, 1630202, 1629675) are entirely excluded from training and used only for testing, so the model is evaluated on players it has never seen — not on unseen games from players it already knows. 968 training rows, 605 test rows.

## Label

A game counts as "extended minutes" if the player played at least 10% more than their trailing 10-game rolling average:

label = 1 if MIN > rolling_avg_min * 1.10 else 0


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

## Model & Evaluation

Logistic regression (scikit-learn) with `class_weight='balanced'`; features standardized with `StandardScaler` fit on the training set only.

Real held-out test set results (605 games, 3 unseen players):
          precision    recall  f1-score   support
       0       0.75      0.60      0.67       401
       1       0.44      0.62      0.51       204
accuracy                           0.60       605

Confusion matrix:

[[239 162]
[ 78 126]]


**Honest read of this result:** the baseline logistic regression was uselessly conservative — 0.18 recall, almost never predicting "yes." Reweighting the loss with `class_weight='balanced'` turned it into an actual detector: recall rose to 0.62 (catching 126 of 204 real extended-minutes games) at the cost of precision falling to 0.44 (162 false alarms). This tradeoff was deliberate — for this audience, missing a real extended-minutes game is worse than a false alarm, which is cheap to sanity-check against the public injury report. But the precision ceiling confirms the deeper issue: reweighting shifts *where* the model commits, it can't manufacture separation that four pre-tip-off features don't contain. The real lever is better features, not tuning — see Limitations & Next Steps.

## Abstention Policy

Forcing a yes/no call on every game at the default 0.5 cutoff isn't honest when many of those calls are near coin-flip odds. `policy.py` adds a third state:

probability >= high → commit to "yes"
probability <= low → commit to "no"
otherwise → abstain — not enough signal, no call


This is written in plain functions (no classes, minimal dependencies) so the logic is auditable line by line — coverage, precision, and recall on the committed cases are computed directly with loops and counters, not a metrics library.

Real calibration sweep on the held-out test set:
low high coverage games precision recall

0.5 0.5 1.0 605 0.438 0.618
0.4 0.6 0.461 279 0.486 0.742
0.35 0.65 0.251 152 0.529 0.776
0.3 0.7 0.136 82 0.566 0.882
0.25 0.75 0.06 36 0.600 0.938


Widening the abstention band raises both precision and recall on committed calls: at the widest band the model commits on only 6% of games but catches 94% of the extended-minutes games it does commit to, at 60% precision. The abstention layer works as intended — it trades coverage for confidence, staying silent on the games where its signal is weak and speaking only where it has real separation. The limitation isn't the policy layer; it's that even at maximum confidence, four pre-tip-off features cap precision around 0.60.

## Engineering Decisions

- **Held out by player, not by row:** a random row-level split would let the model see other games from the same player it's tested on, overstating real-world performance. Holding out entire players is a harder, more honest test of generalization.
- **`class_weight='balanced'` over default weighting:** the unweighted baseline was too conservative to be useful (0.18 recall). Reweighting prioritized recall — appropriate because the target user needs to catch extended-minutes games, and a false "yes" is cheap to verify against the injury report. This raised recall to 0.62 while confirming the precision ceiling is a feature-signal limit, not a tuning one.
- **Logistic regression over a more complex model:** chosen for interpretable, well-calibrated probabilities, which the abstention layer depends on directly. A more complex model might raise accuracy but would make the probability outputs harder to reason about for a threshold-based decision system.
- **Abstention over a forced threshold:** given the cost asymmetry for this audience, silently forcing a call at 0.5 would misrepresent confidence the model doesn't have. Abstaining is the more honest design, even though it reduces how often the model speaks.
- **Plain-function policy layer:** written without classes or a metrics library so every precision/recall number in the calibration table traces back to a simple loop, not a black-box call.

## Limitations

- Only 9 players and 3 seasons — a small, non-representative sample. Results likely don't generalize to the full league without a much larger player pool.
- The four current features capture recent workload and schedule, but nothing about *why* a role would expand — teammate injury/inactive status, foul trouble, or blowout margin aren't modeled, despite being the most likely real drivers of extended minutes.
- Positive-class performance is modest (0.62 recall at 0.44 precision) and precision caps near 0.60 even at maximum model confidence — a signal/feature ceiling, not a tuning problem.
- The serving layer (API, Docker, logging) still reflects the v1 synthetic-data prototype and has not been updated to serve this model.

## Next Steps

- Add public injury/inactive status for the player's team as a feature — likely the single highest-leverage addition, and unlike other candidate features, it's information the target audience actually has pre-game.
- Reconsider the label itself: rather than a per-player binary, modeling *which* of several eligible bench candidates absorbs open minutes, and by how much, may better match what this audience needs.
- Migrate the FastAPI/Docker/logging layer to serve this model and feature set, replacing the v1 synthetic-data version.
- Expand the player/season sample before treating any result here as representative.

## Running the Code

pip install -r requirements.txt # nba_api, pandas, scikit-learn, numpy

python build_features.py # pulls data via nba_api, writes features.csv
<<<<<<< HEAD
python train.py # trains model, prints evaluation + abstention calibration table
=======
python train.py # trains model, prints evaluation + abstention calibration table
>>>>>>> 2514373d3248dcc1d79c4c2ab2e6e79dc2595743
