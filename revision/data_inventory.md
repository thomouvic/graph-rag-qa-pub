# Data Inventory: which CSV is which configuration

## The trap (read first)

The pipeline writes `--graph-compress` (GW) and non-GW runs to the **same directories** with the **same filename pattern**, distinguished only by timestamp and by the `context_chars` column inside the CSV. Filenames look identical:

```
sparql_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260220_200659.csv  # no-GW
sparql_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260220_203018.csv  # +GW (24 min later)
```

Picking the wrong one silently mixes apples and oranges. **Always check `context_chars` before using a baseline/SPARQL CSV.**

## Disambiguation rule

| Configuration | Mean `context_chars` | Approx tokens |
|---|---|---|
| no-GW (full retrieved context) | ~40-44k | ~10k |
| +GW (graph-walk compressed) | ~19k | ~5k |

Threshold: `context_chars < 25000` → GW. Otherwise no-GW.

## Full inventory (all 500-question CSVs, large_scale)

### HotpotQA — `experiments/hotpotqa/large_scale/`

| Method | Model | Cfg | Acc | Abs | ctx_avg | Timestamp |
|---|---|---|---|---|---|---|
| baseline | 8B | no-GW | 67.0% | 12.2% | 43657 | 20260218_101626 |
| baseline | 8B | +GW | 63.6% | 19.8% | 19313 | 20260220_170058 |
| baseline | 70B | no-GW | 78.0% | 7.8% | 43657 | 20260218_110034 |
| baseline | 70B | +GW | 79.2% | 6.2% | 19313 | 20260220_171051 |
| sparql | 8B | no-GW | 70.6% | 5.8% | 43657 | 20260218_103802 |
| sparql | 8B | +GW | 76.6% | 2.0% | 19313 | 20260220_171837 |
| sparql | 70B | no-GW | 80.2% | 8.2% | 43657 | 20260218_111824 |
| sparql | 70B | +GW | 79.6% | 9.0% | 19313 | 20260220_172852 |
| self_ask | 8B | no-GW | 57.0% | 29.8% | 43657 | 20260429_141058 |
| self_ask | 70B | no-GW | 70.0% | 20.0% | 43657 | 20260429_143336 |
| self_ask_maxtok1000 | 8B | no-GW | 56.4% | 29.6% | 43657 | 20260430_110509 |
| self_ask_maxtok1000 | 70B | no-GW | 70.4% | 20.6% | 43657 | 20260430_112754 |

### 2WikiMultiHopQA — `experiments/2wikimultihopqa/large_scale/`

| Method | Model | Cfg | Acc | Abs | ctx_avg | Timestamp |
|---|---|---|---|---|---|---|
| baseline | 8B | no-GW | 31.4% | 46.0% | 39451 | 20260220_193514 |
| baseline | 8B | +GW | 30.4% | 54.4% | 19087 | 20260220_195654 |
| baseline | 70B | no-GW | 48.8% | 34.8% | 39451 | 20260220_204350 |
| baseline | 70B | +GW | 53.6% | 25.0% | 19087 | 20260220_210131 |
| sparql | 8B | no-GW | 45.6% | 21.6% | 39451 | 20260220_200659 |
| sparql | 8B | +GW | 55.8% | 15.6% | 19087 | 20260220_203018 |
| sparql | 70B | no-GW | 61.0% | 30.6% | 39451 | 20260220_211014 |
| sparql | 70B | +GW | 59.8% | 32.2% | 19087 | 20260220_212846 |
| generic_cot | 8B | no-GW | 52.6% | 25.2% | 39451 | 20260221_140052 |
| generic_cot | 70B | no-GW | 56.4% | 36.8% | 39451 | 20260221_142306 |
| self_ask | 8B | no-GW | 42.2% | 42.6% | 39451 | 20260429_172708 |
| self_ask | 70B | no-GW | 58.6% | 34.4% | 39451 | 20260429_174943 |
| self_ask_maxtok1000 | 8B | no-GW | 42.6% | 39.8% | 39451 | 20260430_115140 |
| self_ask_maxtok1000 | 70B | no-GW | 56.8% | 36.8% | 39451 | 20260430_121413 |

### MuSiQue — `experiments/musique/large_scale/`

| Method | Model | Cfg | Acc | Abs | ctx_avg | Timestamp |
|---|---|---|---|---|---|---|
| baseline | 8B | no-GW | 23.6% | 52.0% | 44762 | 20260220_085621 |
| baseline | 8B | +GW | 19.4% | 60.6% | 19733 | 20260220_091804 |
| baseline | 70B | no-GW | 35.2% | 42.4% | 44762 | 20260220_073938 |
| baseline | 70B | +GW | 39.8% | 34.6% | 19733 | 20260220_080124 |
| sparql | 8B | no-GW | 28.8% | 20.6% | 44762 | 20260220_092744 |
| sparql | 8B | +GW | 30.6% | 13.4% | 19733 | 20260220_095057 |
| sparql | 70B | no-GW | 43.8% | 34.2% | 44762 | 20260220_080936 |
| sparql | 70B | +GW | 42.2% | 38.2% | 19733 | 20260220_082837 |
| self_ask | 8B | no-GW | 43.4% | 18.2% | 44762 | 20260429_191237 |
| self_ask | 70B | no-GW | 47.4% | 26.6% | 44762 | 20260429_193542 |
| self_ask_maxtok1000 | 8B | no-GW | 43.2% | 18.8% | 44762 | 20260430_123325 |
| self_ask_maxtok1000 | 70B | no-GW | 48.0% | 26.0% | 44762 | 20260430_125622 |
| self_ask_demos-2wikimultihopqa | 8B | no-GW | 25.0% | 48.6% | 44762 | 20260430_000215 |
| self_ask_demos-2wikimultihopqa | 70B | no-GW | 46.8% | 30.6% | 44762 | 20260429_234208 |
| sparql_musique_tuned | 8B | no-GW | 38.0% | 6.2% | 44762 | 20260429_223112 |
| sparql_musique_tuned | 70B | no-GW | 46.6% | 28.2% | 44762 | 20260429_221114 |

## Notes

- **GW ambiguity only affects baseline and SPARQL CoT** — both have one no-GW and one +GW run per (dataset, model). Other methods (`generic_cot`, `self_ask*`, `sparql_musique_tuned`) were always run on full no-GW context; their CSVs have no GW counterpart.
- **HotpotQA pattern is special**: no-GW runs are on Feb 18, +GW on Feb 20. For 2WikiMHQA and MuSiQue, both runs are on Feb 20 with the no-GW being the *earlier* same-day timestamp and +GW being ~20 min later.
- The self-ask `_maxtok1000` variant runs use `max_tokens=1000` (vs default 2048); see `revision/self_ask_results.md` cost section.
- The `self_ask_demos-2wikimultihopqa` variant on MuSiQue is the "untuned Self-Ask" experiment (Press et al.'s 2WikiMHQA prompt transferred to MuSiQue per their Bamboogle precedent).
- `sparql_musique_tuned` is the MuSiQue-tuned-prompt variant of SPARQL CoT.

## Cross-references

- Headline comparison table (no-GW SPARQL vs Self-Ask): `revision/self_ask_results.md`.
- Coverage analysis using paper-era no-GW baselines: `revision/coverage/summary.txt`.
- Underlying script that picks paper-era runs: `find_coverage.py` and `supporting_fact_coverage.py` use os.listdir + pattern; **they currently pick the latest by filename, which for HotpotQA is the +GW run** — flagged for cleanup if anyone touches them.

## Convention going forward

When adding new runs:
- If the run uses GW compression, ideally tag the output filename with `_gw_` or write to a separate `results_*_gw_groq/` dir. (Current `run.py` does not do this; future fix worth ~15 min.)
- When loading existing CSVs in any analysis script, **filter on `context_chars` not just filename** to disambiguate GW from no-GW.
