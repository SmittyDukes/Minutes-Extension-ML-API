
"""
Abstention policy - simple version.

What problem are we solving?
-----------------------------
Our model spits out a probability, like 0.42, meaning "I think there's a 42%
chance this game has extended minutes." Normally you'd just say: if it's
above 0.5, call it "yes", otherwise call it "no". One cutoff, two outcomes.

The problem: our model isn't very good yet. We measured it and found that
even when it says "yes", it's only right about 55% of the time. That's
barely better than a coin flip. And a "yes" here isn't free - it makes the
coach actually change the rotation and pull minutes from another player.
If the model is wrong, that's a real cost.

So instead of forcing a yes/no answer every time, we add a third option:
"I'm not sure." If the probability is stuck in the middle (not clearly
low, not clearly high), we just say so instead of guessing.


How it works
------------
We pick two numbers, low and high. For example: low = 0.30, high = 0.70

    - probability <= 0.30  -> we say "no"      (confident enough)
    - probability >= 0.70  -> we say "yes"     (confident enough)
    - anything in between  -> we say "abstain" (not confident, no call)

Everything below is just plain functions. No classes, no fancy libraries.
"""

import random


def decide(probability, low=0.30, high=0.70):
    """
    Take one probability and turn it into a decision.

    Returns a dictionary with:
        - "label": "yes", "no", or "abstain"
        - "message": something human-readable to show the coach
    """

    if probability >= high:
        label = "yes"
        message = "Likely extended minutes. Confident enough to act on."
    elif probability <= low:
        label = "no"
        message = "Unlikely extended minutes. Confident enough to act on."
    else:
        label = "abstain"
        message = "Not enough signal for this game. No rotation call - treat as normal."

    return {
        "probability": probability,
        "label": label,
        "message": message,
    }


def decide_many(probabilities, low=0.30, high=0.70):
    """Run decide() on a whole list of probabilities and return a list of results."""
    results = []
    for p in probabilities:
        results.append(decide(p, low, high))
    return results


def coverage(probabilities, low=0.30, high=0.70):
    """
    How often is the model willing to actually commit to yes/no,
    instead of abstaining? Returns a number between 0 and 1.

    Example: coverage of 0.47 means the model commits on 47% of games
    and abstains on the other 53%.
    """
    committed_count = 0
    for p in probabilities:
        if p >= high or p <= low:
            committed_count += 1

    if len(probabilities) == 0:
        return 0.0
    return committed_count / len(probabilities)


def check_one_threshold_pair(y_true, y_prob, low, high):
    """
    Given real answers (y_true: 0 or 1) and the model's predicted
    probabilities (y_prob), check how good the model is IF we only look
    at the games where it was willing to commit (not abstaining).

    This is the key idea: as you make "low" smaller and "high" bigger,
    the model abstains more often, but should be MORE correct on the
    games it does commit to. This function measures that tradeoff.
    """

    true_positives = 0  # model said yes, and it really was yes
    false_positives = 0  # model said yes, but it was actually no
    false_negatives = 0  # model said no, but it was actually yes
    committed_count = 0  # games where the model didn't abstain

    for actual, prob in zip(y_true, y_prob):
        if prob >= high:
            # model commits to "yes"
            committed_count += 1
            if actual == 1:
                true_positives += 1
            else:
                false_positives += 1
        elif prob <= low:
            # model commits to "no"
            committed_count += 1
            if actual == 1:
                false_negatives += 1
            # if actual == 0, model correctly said no - not needed for the math below
        # else: probability is between low and high, model abstains, skip it

    total_games = len(y_true)
    coverage_rate = committed_count / total_games if total_games else 0

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = None  # model never said "yes" in this band

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = None

    return {
        "low": low,
        "high": high,
        "coverage": round(coverage_rate, 3),
        "games_committed": committed_count,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
    }


def print_threshold_table(y_true, y_prob, threshold_pairs):
    """
    Try a bunch of (low, high) pairs and print a table so you can see
    the tradeoff: wider abstention band = higher precision, lower coverage.
    """
    print(f"{'low':>5} {'high':>5} {'coverage':>9} {'games':>6} {'precision':>10} {'recall':>7}")
    print("-" * 50)

    for low, high in threshold_pairs:
        result = check_one_threshold_pair(y_true, y_prob, low, high)
        print(
            f"{result['low']:>5} {result['high']:>5} "
            f"{result['coverage']:>9} {result['games_committed']:>6} "
            f"{str(result['precision']):>10} {str(result['recall']):>7}"
        )


# ---------------------------------------------------------------------------
# Demo: fake data just to prove the functions work before you plug in the
# real model's predictions.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)

    # Make some fake probabilities and fake true answers, similar in size to
    # your real data (204 real "yes" games, 371 real "no" games).
    y_true = []
    y_prob = []

    for _ in range(371):
        y_true.append(0)
        y_prob.append(max(0, min(1, random.gauss(0.30, 0.18))))

    for _ in range(204):
        y_true.append(1)
        y_prob.append(max(0, min(1, random.gauss(0.50, 0.20))))

    # Try one single game
    example = decide(0.42)
    print("Example decision for probability 0.42:")
    print(f"  label: {example['label']}")
    print(f"  message: {example['message']}")
    print()

    # Check overall coverage at one threshold pair
    print(f"Coverage at (0.30, 0.70): {coverage(y_prob, 0.30, 0.70):.2%}")
    print()

    # Try several threshold pairs and compare
    pairs_to_try = [
        (0.50, 0.50),  # no abstention at all, same as your current model
        (0.40, 0.60),
        (0.35, 0.65),
        (0.30, 0.70),
        (0.25, 0.75),
    ]
    print("Comparing threshold pairs:")
    print_threshold_table(y_true, y_prob, pairs_to_try)
