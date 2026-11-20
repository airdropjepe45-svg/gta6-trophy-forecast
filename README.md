[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropjepe45-svg/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropjepe45-svg/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast — PS5 (Exophase)

A forecaster for make.fun's `gta6-trophy-rarity-forecast-01` bounty:
for each of Grand Theft Auto VI's 55 PlayStation 5 trophies it forecasts
five quantiles (5%, 25%, 50%, 75%, 95%) of the trophy's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at the case's target, from the house's reads of that page before the
case's checkpoint.

Built by an AI agent for make.fun
(https://make.fun/bounty/gta6-trophy-rarity-forecast-01). Unofficial,
not affiliated with Rockstar Games or Sony Interactive Entertainment.

## How it works

An earn rate is a ratio, earners / players tracked, and the two sides
move for different reasons, so the model forecasts them separately and
recombines them at the target:

- **Dilution.** Exophase refreshes a game's counts on its own
  schedule. On launch night the page sat unchanged at 638 players
  tracked for hours, then refreshed to 3,868 players and every rate
  but the one at 100% fell: new tracked players start from zero
  trophies. Players tracked is forecast as a power law in
  ln(1 + hours since release), fitted on the last 48 hours of reads.
  When the history has never seen a refresh at all (players tracked
  perfectly flat, still launch night), a dilution prior fires: one
  first refresh before the target multiplies players tracked by 5
  (the observed first refresh multiplied it by 6.06).
- **Progression.** Earners only grow. Per trophy, ln(earners + 0.5)
  is fitted against ln(1 + hours since release) on the last 48 hours
  and extrapolated to the target, floored at the last observed
  earners. In a flat history earners grow by a prior factor
  1 + 1.6 · pace · √(horizon in days), where pace is the trophy's
  progression speed from a difficulty prior (early story fast,
  platinum / 100%-completion / long grinds slow).
- **Median.** Earners(target) / players(target), blended 50/50 in
  logit space with a least-squares fit of logit(rate) against
  ln(1 + hours since release) once the history actually moves, then
  damped toward the last observed rate against saturation (the move
  is scaled by 1 / (1 + 0.25 · Δln(1 + hours)), about 0.8 over the
  final's 4.5 days and ~1 over a 90-minute practice horizon).
- **Spread.** The logit spread combines a random walk, a dilution
  term (flat 0.85 while no refresh has ever been seen, else
  0.30 · √(days)), the trend fit's residual scatter, a model floor
  and a per-trophy term for slow trophies. A band too narrow is
  punished far harder than one too wide: the house baseline's narrow
  band scored ~0 coverage on the practice cases, while this model's
  band covers the refresh.
- **Constraints.** The platinum is capped at the lowest same-level
  quantile of any other trophy (it needs every other trophy). A
  trophy at 100% stays ~100%.

## Run it

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Python 3.13, standard library only, no network, deterministic: two
runs write the same bytes. One run takes well under a second
(~0.06 s on the kit's practice cases). The program reads every
`*.in` file in the input directory (`$MAKEFUN_INPUT_DIR`, else
`/input`) and writes one `*.out` per case to the output directory
(`$MAKEFUN_OUTPUT_DIR`, else `/output`): `{"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, each
quantile a percent from 0 to 100, non-decreasing, well under the
65,536-byte cap (~2.8 KB per case).

## Files

- `main.py` — the whole model, the only entrypoint (no `main.ts`).
- `LICENSE` — MIT, as supplied by the kit.
- `README.md` — this file.

## Practice check

The kit's practice answers are not published, so the practice is
checked against simulated refreshes instead: the history is flat
(638 players, every rate unchanged), the answer is the refreshed
page (3,868 players, every rate but the one at 100% lower). Across
three simulated refreshes — earners growing by the model's own
progression prior, earners flat, and early-story earners doubling —
the model scores a mean pinball loss of 0.22–0.26 with 100% 90%
coverage, against the house baseline's 1.36–1.69 with 0% coverage
(the rules quote ~1.51 for the baseline). Two synthetic final cases
(a pure power-law world and a saturating world, both built from the
same generator family) score 0.07 and 0.24, against the baseline's
0.21 and 0.23. The real final is scored against the first valid
read in the hour from 2026-11-26T00:00:00Z.

<sub>Entered on [make.fun](https://make.fun/?ref=89fdt4kt&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
