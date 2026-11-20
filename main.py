"""Forecast GTA VI's PS5 trophy earn rates on Exophase at a case's target.

Program for make.fun bounty gta6-trophy-rarity-forecast-01. Python 3.13,
standard library only, deterministic, no network.

The model
---------
An earn rate is a ratio, earners / players_tracked, and the two sides move
for different reasons, so they are forecast separately and the rate is
recombined at the target:

* DILUTION. Exophase refreshes a game's counts on its own schedule, not
  with every player's trophy. On launch night the page sat unchanged at
  638 players tracked for hours, then refreshed to 3,868 players, and
  every rate but the one at 100% fell: new tracked players start from
  zero trophies, so a jump in players_tracked drags every rate down.
  players_tracked is therefore forecast on its own, as a power law in
  ln(1 + hours since release), fitted on the case's last 48 hours of
  reads. When the history has never seen a refresh at all (players
  tracked perfectly flat, still launch night), a dilution prior fires:
  one first refresh before the target multiplies players tracked by 5
  (the observed first refresh multiplied it by 6.06; 5 is the
  conservative side of that). That prior only ever fires on a flat
  launch-night history; once players have moved, the data rules.

* PROGRESSION. Earners only grow: players keep playing and earning.
  Per trophy, ln(earners + 0.5) is fitted against ln(1 + hours since
  release) on the last 48 hours of reads and extrapolated to the
  target, floored at the last observed earners and capped at the
  forecast players. When the history is flat, earners instead grow by
  a prior factor 1 + 1.6 * pace * sqrt(horizon in days), where pace is
  the trophy's progression speed from the difficulty prior below:
  early story trophies are earned fast, 100%-completion and grind
  trophies slowly.

* MEDIAN. The median rate is earners(target) / players(target). Where
  the history actually moves, it is blended 50/50 in logit space with
  a straight least-squares fit of logit(rate) against ln(1 + hours
  since release); where the history is flat that fit says nothing (it
  would just repeat the stale rate), so the numerator/denominator
  forecast stands alone. Both fits are power laws in ln(1 + hours),
  which keep growing where real adoption saturates, so the blended
  median is damped toward the last observed rate: the move is scaled
  by 1 / (1 + 0.25 * (x_target - x_last)), which is ~0.8 over the
  final's 4.5 days and ~1 over a 90-minute practice horizon. The
  dilution prior in a flat history is a discrete jump, not a trend,
  so it is never damped.

* DIFFICULTY PRIOR. pace per trophy (0..1) comes from the trophy's
  own difficulty: story-chapter trophies progress fast, the platinum
  and the 100%-completion trophy slowest, long grinds (250 stunts,
  500 choices, 5 bank vaults, gold medals at every shooting range)
  slow. The table is built from the public trophy guide research
  (chapter order, counts, grades); unknown ids fall back to a formula
  from tier, points and description.

* SPREAD. The logit spread is the root sum of squares of: a random
  walk 0.35 * sqrt(days); a dilution term, 0.30 * sqrt(days) once
  refreshes are observed, or a flat 0.85 while no refresh has ever
  been seen (a refresh of unknown size is the dominant risk, and the
  house baseline's narrow band scored ~0 coverage on the practice
  exactly because it missed it); the residual scatter of the logit
  trend fit; a 0.25 model floor; and a small per-trophy term for slow
  trophies. The spread errs wide, but not wildly so: a band far wider
  than the truth costs at every trophy, while one too narrow pays
  hard only for the trophies outside it, so the components above are
  sized to the scatter the history itself shows.

* CONSTRAINTS. The platinum can only be earned by players who earned
  every other trophy, so each of its quantiles is capped at the
  lowest same-level quantile of any other trophy. A trophy at 100%
  in the last read stays ~100% (the opening-mission trophy stayed at
  100% through the 6x refresh), forecast as [99.9, 99.97, 100, 100,
  100]. Rates are clamped to 0.01..99.99 before any logit.

The program reads every *.in file in the input directory and writes
one *.out per case, so two runs write the same bytes. It runs in well
under the 8-second limit: 55 trophies, a few hundred reads at most,
all closed-form least squares.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Release of GTA VI: the anchor for ln(1 + hours since release).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# How far back the trend fits look (hours before the case's last read):
# 48 hours. A shorter window would track saturation better, but the
# page prints rates to 2 decimals, so on rare trophies the rounding
# noise swamps a short window's slope; 48 hours of 10-minute reads
# beats that noise down while still being recent growth.
FIT_WINDOW_HOURS = 48.0
# Damping of the extrapolated move toward the last observed rate, per
# unit of ln(1 + hours) between the last read and the target: adoption
# saturates, so a power law fitted on recent growth overshoots.
DAMP_COEF = 0.25
# The dilution prior: first-refresh multiplier for players tracked when
# the history has never seen a refresh (flat launch night).
FIRST_REFRESH_JUMP = 5.0
# Earners' prior growth per sqrt(day) in a flat history, scaled by pace.
PROGRESSION_RATE = 1.6
# Spread components, in logit units (see the module docstring).
SIGMA_RW = 0.35          # random walk, per sqrt(day)
SIGMA_DILUTION = 0.30    # dilution uncertainty, per sqrt(day)
SIGMA_DILUTION_FLAT = 0.85  # one unobserved refresh of unknown size
SIGMA_MODEL = 0.25       # floor for everything the components miss
SIGMA_PACE = 0.15        # extra for slow trophies (1 - pace)

# Progression speed per trophy id (0..1), from the public trophy guide:
# story order, counts and grades. Early story is fast; the platinum,
# 100% completion and the long grinds are slow.
PACE = {
    1: 0.08,   # platinum: needs all 54 others
    2: 1.00,   # Gone Fishing, opening mission (already 100%)
    3: 0.85,   # Chapter 1
    4: 0.75,   # Chapter 2
    5: 0.62,   # Chapter 3
    6: 0.50,   # Chapter 4
    7: 0.35,   # final mission (gold)
    8: 0.80,   # Breakfast Run
    9: 0.70,   # The Cartel Heist
    10: 0.75,  # Child's Play
    11: 0.70,  # Headliners
    12: 0.65,  # VCIA Takedown
    13: 0.55,  # Operation Savage Glory
    14: 0.12,  # 100% Completion (gold, ~100h cleanup)
    15: 0.35,  # 250 stunts (silver)
    16: 0.65,  # try each activity once
    17: 0.25,  # gold in all 42 Shooting Range challenges (silver)
    18: 0.45,  # win 14 drone races
    19: 0.50,  # win 20 races
    20: 0.60,  # beat Cal fishing
    21: 0.70,  # complete an arcade game
    22: 0.60,  # Andy strand finale
    23: 0.60,  # Kenton strand finale
    24: 0.60,  # Billy strand finale
    25: 0.40,  # 8 stranger strands (silver)
    26: 0.60,  # Randy strand finale
    27: 0.60,  # Tyrone strand finale
    28: 0.45,  # meditate at 18 spots (the one missable trophy)
    29: 0.60,  # hunt a unique animal for Bret
    30: 0.60,  # 3 card sets (silver)
    31: 0.40,  # WAiNK at 100 locations (silver)
    32: 0.45,  # 50 challenges
    33: 0.60,  # lose a 4-star wanted level with partner
    34: 0.55,  # rob 3 shops back-to-back
    35: 0.60,  # discover 100 vehicles
    36: 0.65,  # 5 takedowns in one Combat Focus
    37: 0.65,  # unscheduled departure at VCIA
    38: 0.60,  # Rhino tank
    39: 0.50,  # rob a jewelry store undetected
    40: 0.30,  # rob 5 bank vaults (silver)
    41: 0.65,  # rob an armored truck
    42: 0.55,  # explore all of Leonida
    43: 0.35,  # rob an operation of each of 9 gangs
    44: 0.70,  # cocktail on Ocean Beach
    45: 0.65,  # smash out of a vehicle showroom
    46: 0.65,  # bowling follow-through
    47: 0.70,  # partnership story beat
    48: 0.20,  # 500 choices
    49: 0.70,  # second time's the charm
    50: 0.60,  # story oddity
    51: 0.55,  # melee drive-by bowling
    52: 0.30,  # launder $250,000
    53: 0.35,  # spend $100,000
    54: 0.30,  # buy all garage properties
    55: 0.40,  # register 10 vehicles
}

# Tier fallback for a trophy with no history at all (never expected).
TIER_PRIOR_RATE = {"platinum": 0.05, "gold": 0.5, "silver": 2.0, "bronze": 6.0}


def parse_ts(text):
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as the starter does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"level {p} is not in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def ols(xs, ys):
    """Least squares y = a + b*x; returns (a, b, rmse) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    rmse = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return (a, b, rmse)


def pace_for(trophy):
    """Progression speed of a trophy: the table first, else tier/description."""
    pace = PACE.get(trophy.get("id"))
    if pace is not None:
        return pace
    desc = str(trophy.get("description") or "").lower()
    tier = str(trophy.get("tier") or "bronze").lower()
    if "100% completion" in desc or tier == "platinum":
        return 0.10
    if tier == "gold":
        pace = 0.35
    elif tier == "silver":
        pace = 0.45
    else:
        pace = 0.60
    if "chapter 1" in desc:
        pace = max(pace, 0.85)
    elif "chapter 2" in desc:
        pace = max(pace, 0.75)
    elif "chapter 3" in desc:
        pace = max(pace, 0.62)
    elif "chapter 4" in desc:
        pace = max(pace, 0.50)
    return pace


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    if len(levels) != 5:
        raise ValueError(f"{name} asks for {len(levels)} levels, not 5")
    z = [norm_ppf(float(tau)) for tau in levels]
    history = case.get("history") or []
    trophies = case["trophies"]
    horizon_days = float(case["horizon_minutes"]) / 1440.0

    try:
        release = parse_ts(case["game"]["release_date"])
    except (KeyError, TypeError, ValueError):
        release = RELEASE
    if release.tzinfo is None:
        release = release.replace(tzinfo=timezone.utc)
    try:
        checkpoint = parse_ts(case["at"])
        target = parse_ts(case["target"])
    except (KeyError, TypeError, ValueError):
        checkpoint = target = None

    if not history:
        # No reads at all: fall back to the tier prior, very wide.
        rows = []
        for trophy in trophies:
            med = TIER_PRIOR_RATE.get(str(trophy.get("tier") or "").lower(), 5.0)
            lg = logit(med)
            rows.append({"id": trophy["id"],
                         "q": [round4(min(100.0, max(0.01, inv_logit(lg + zz * 2.0))))
                               for zz in z]})
        return {"trophies": rows}

    # --- Parse the reads: (x, players, [rates]) with x = ln(1 + hours since release).
    reads = []
    for read in history:
        try:
            at = parse_ts(read["at"])
            players = float(read["players_tracked"])
        except (KeyError, TypeError, ValueError):
            continue
        if players <= 0:
            continue
        hours = max(0.0, (at - release).total_seconds() / 3600.0)
        rates = read.get("rates") or []
        reads.append((math.log1p(hours), players, rates, at))
    if not reads:
        raise ValueError(f"{name} has no usable reads")
    reads.sort(key=lambda r: r[3])
    x_last = reads[-1][0]
    players_last = reads[-1][1]
    last_rates = reads[-1][2]
    try:
        x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
    except (TypeError, AttributeError):
        x_target = x_last + math.log1p(horizon_days * 24.0)

    # Fit window: the last 48 hours of reads (all of them if fewer).
    window_start = reads[-1][3].timestamp() - FIT_WINDOW_HOURS * 3600.0
    window = [r for r in reads if r[3].timestamp() >= window_start]
    if len(window) < 3:
        window = reads

    players_seen = [r[1] for r in reads]
    players_flat = max(players_seen) == min(players_seen)
    launch_night = checkpoint is not None and \
        (checkpoint - release).total_seconds() <= 36 * 3600.0

    # --- Denominator: players tracked at the target.
    if players_flat and launch_night:
        # No refresh has ever been seen: the first refresh lands before
        # the target and multiplies tracked players (observed x6.06).
        players_target = players_last * FIRST_REFRESH_JUMP
    else:
        fit = ols([r[0] for r in window], [math.log(r[1]) for r in window])
        if fit is None:
            players_target = players_last
        else:
            a, b, _ = fit
            players_target = math.exp(a + max(0.0, b) * x_target)
        players_target = min(max(players_target, players_last), players_last * 10.0)

    # --- Per trophy: numerator, trend, median, spread.
    rows = []
    platinum_idx = None
    for i, trophy in enumerate(trophies):
        if str(trophy.get("tier") or "").lower() == "platinum":
            platinum_idx = len(rows)
        pace = pace_for(trophy)

        # Series for this trophy over the window (and over all reads for
        # the flat check): earners and logit rates.
        earners_w, earners_all = [], []
        logits_w = []
        for r in window:
            rate = r[2][i] if i < len(r[2]) else None
            if rate is None:
                continue
            earners_w.append((r[0], float(rate) / 100.0 * r[1]))
            logits_w.append((r[0], logit(float(rate))))
        for r in reads:
            rate = r[2][i] if i < len(r[2]) else None
            if rate is not None:
                earners_all.append(float(rate) / 100.0 * r[1])
        last_rate = last_rates[i] if i < len(last_rates) else None

        # A trophy sitting at 100% stays ~100%: everyone tracked has
        # earned the opening mission, through the 6x refresh as well.
        if last_rate is not None and float(last_rate) >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [99.9 if tau <= 0.05 else 99.97 if tau <= 0.25 else 100
                               for tau in levels]})
            continue

        if not earners_w:
            # Never observed: tier prior, very wide.
            lg = logit(TIER_PRIOR_RATE.get(str(trophy.get("tier") or "").lower(), 5.0))
            rows.append({"id": trophy["id"],
                         "q": [round4(min(100.0, max(0.01, inv_logit(lg + zz * 2.0))))
                               for zz in z]})
            continue

        earners_last = earners_w[-1][1]
        earners_flat = earners_all and max(earners_all) == min(earners_all)

        # Numerator: earners at the target.
        if earners_flat:
            prog = 1.0 + PROGRESSION_RATE * pace * math.sqrt(horizon_days)
            earners_target = earners_last * prog
            if earners_last <= 0:
                earners_target = 0.5 * prog
        else:
            fit_e = ols([x for x, _ in earners_w],
                        [math.log(max(e, 0.0) + 0.5) for _, e in earners_w])
            if fit_e is None:
                earners_target = earners_last
            else:
                a, b, _ = fit_e
                earners_target = math.exp(a + max(0.0, b) * x_target) - 0.5
            earners_target = max(earners_target, earners_last, 0.5)
        earners_target = min(earners_target, players_target)
        rate_ep = 100.0 * earners_target / players_target

        # Trend: logit(rate) against ln(1 + hours since release).
        fit_l = ols([x for x, _ in logits_w], [y for _, y in logits_w]) \
            if len(logits_w) >= 3 else None
        resid = 0.15
        if fit_l is not None:
            a, b, rmse = fit_l
            resid = min(max(rmse, 0.15), 0.50)
            logit_trend = a + b * x_target
        else:
            logit_trend = logit(last_rate) if last_rate is not None else logit(rate_ep)

        # Median: the numerator/denominator forecast alone while the
        # history is flat (the trend would just repeat the stale rate);
        # blended with the trend once the history actually moves, then
        # damped toward the last observed rate against saturation (the
        # flat-history dilution jump is discrete and is never damped).
        if players_flat:
            median_logit = logit(rate_ep)
        else:
            median_logit = 0.5 * logit(rate_ep) + 0.5 * logit_trend
            if last_rate is not None:
                damp = 1.0 / (1.0 + DAMP_COEF * max(0.0, x_target - x_last))
                median_logit = logit(last_rate) + damp * (median_logit - logit(last_rate))

        # Spread: random walk + dilution + trend residual + model floor
        # + a little extra for slow trophies.
        sigma_rw = SIGMA_RW * math.sqrt(horizon_days)
        sigma_dil = SIGMA_DILUTION_FLAT if players_flat \
            else SIGMA_DILUTION * math.sqrt(horizon_days)
        sigma = math.sqrt(sigma_rw ** 2 + sigma_dil ** 2 + resid ** 2
                          + SIGMA_MODEL ** 2 + (SIGMA_PACE * (1.0 - pace)) ** 2)

        qs = [round4(min(100.0, max(0.01, inv_logit(median_logit + zz * sigma))))
              for zz in z]
        # Rounding keeps the order, but enforce it explicitly.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    # The platinum is earned only by players who earned every other
    # trophy, so cap each of its quantiles at the lowest same-level
    # quantile of any other trophy.
    if platinum_idx is not None and len(rows) > 1:
        plat = rows[platinum_idx]
        for k in range(5):
            cap = min(row["q"][k] for j, row in enumerate(rows) if j != platinum_idx)
            if plat["q"][k] > cap:
                plat["q"][k] = cap
        for k in range(1, 5):
            if plat["q"][k] < plat["q"][k - 1]:
                plat["q"][k] = plat["q"][k - 1]

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
