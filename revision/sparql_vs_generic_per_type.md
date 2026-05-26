# Per-type SPARQL CoT vs Generic CoT (eQxk #5, second half)

Addresses the second part of eQxk #5: *"Tables 9 and 10 decompose results by question type across all three datasets, but they compare SPARQL CoT to baseline and GW to non-GW — not SPARQL CoT to generic CoT."*

This is the SPARQL-vs-Generic version of the per-type tables, on each dataset's **native** type taxonomy (no cross-dataset collapsing).

## Scope: this is *not* a routing analysis

These two questions are distinct and should not be confused:

- **Per-type SPARQL vs Generic** (this file): ablation — which method works on which gold type, on each dataset. Descriptive, no routing involved.
- **Routing** ([routing_validation.md](routing_validation.md)): operational deployment of a per-type rule via a classifier. Aggregate accuracy.

Mixing them invites the reader to confuse the rule's *empirical basis* (per-type ablation) with the rule's *deployed performance* (routed accuracy). Separate files keep the logic clean.

**Data**: existing no-GW SPARQL CoT and Generic CoT CSVs (paper-era timestamps per [data_inventory.md](data_inventory.md)). 500 questions per dataset. No new compute.

## HotpotQA (native types: bridge, comparison)

| Model | Type | n | SPARQL Acc | Generic Acc | Gap (S−G) | Winner |
|---|---|---|---|---|---|---|
| 8B | bridge | 398 | 71.1% | 74.9% | -3.8 | Generic |
| 8B | comparison | 102 | 68.6% | 78.4% | -9.8 | Generic |
| 70B | bridge | 398 | 80.2% | 78.4% | +1.8 | SPARQL |
| 70B | comparison | 102 | 80.4% | 85.3% | -4.9 | Generic |

## 2WikiMultiHopQA (4 native types)

| Model | Type | n | SPARQL Acc | Generic Acc | Gap (S−G) | Winner |
|---|---|---|---|---|---|---|
| 8B | compositional | 205 | 41.0% | 35.6% | +5.4 | SPARQL |
| 8B | bridge_comparison | 121 | 37.2% | 46.3% | -9.1 | Generic |
| 8B | comparison | 120 | 66.7% | 84.2% | -17.5 | Generic |
| 8B | inference | 54 | 35.2% | 61.1% | -25.9 | Generic |
| 70B | compositional | 205 | 48.3% | 46.3% | +2.0 | SPARQL |
| 70B | bridge_comparison | 121 | 53.7% | 33.9% | +19.8 | SPARQL |
| 70B | comparison | 120 | 91.7% | 94.2% | -2.5 | Generic |
| 70B | inference | 54 | 57.4% | 61.1% | -3.7 | Generic |

## MuSiQue (native types: hop counts)

| Model | Type | n | SPARQL Acc | Generic Acc | Gap (S−G) | Winner |
|---|---|---|---|---|---|---|
| 8B | 2hop | 264 | 37.5% | 40.2% | -2.7 | Generic |
| 8B | 3hop1 | 110 | 19.1% | 40.0% | -20.9 | Generic |
| 8B | 4hop1 | 60 | 20.0% | 40.0% | -20.0 | Generic |
| 8B | 3hop2 | 36 | 22.2% | 25.0% | -2.8 | Generic |
| 8B | 4hop3 | 17 | 11.8% | 5.9% | +5.9 | SPARQL (small n) |
| 8B | 4hop2 | 13 | 15.4% | 7.7% | +7.7 | SPARQL (small n) |
| 70B | 2hop | 264 | 49.2% | 41.3% | +8.0 | SPARQL |
| 70B | 3hop1 | 110 | 40.9% | 40.0% | +0.9 | SPARQL |
| 70B | 4hop1 | 60 | 41.7% | 40.0% | +1.7 | SPARQL |
| 70B | 3hop2 | 36 | 38.9% | 27.8% | +11.1 | SPARQL |
| 70B | 4hop3 | 17 | 11.8% | 11.8% | 0.0 | tie |
| 70B | 4hop2 | 13 | 23.1% | 15.4% | +7.7 | SPARQL |

## Read

Three patterns emerge.

### Pattern 1: Generic dominates on 8B almost everywhere

| Dataset | 8B native types where Generic wins |
|---|---|
| HotpotQA | both (bridge -3.8, comparison -9.8) |
| 2WikiMHQA | 3 of 4 (bridge_comparison -9.1, comparison -17.5, inference -25.9; only loses compositional +5.4) |
| MuSiQue | 4 of 6 hop counts (loses only on small-n 4hop2 +7.7 and 4hop3 +5.9) |

Generic CoT is the safer default on 8B for nearly any question type, on any dataset.

### Pattern 2: SPARQL dominates on 70B for "structural" types

| Dataset | 70B types where SPARQL wins |
|---|---|
| HotpotQA | bridge (+1.8) — but loses comparison (-4.9) |
| 2WikiMHQA | compositional (+2.0), bridge_comparison (**+19.8** — largest gap in the analysis) |
| MuSiQue | 5 of 6 hop counts (2hop +8.0, 3hop1 +0.9, 4hop1 +1.7, 3hop2 +11.1, 4hop2 +7.7) |

On 70B, SPARQL CoT is the better default for graph-structural / multi-hop types.

### Pattern 3: "comparison" and "inference" types always favor Generic, regardless of model

| Dataset | Model | Type | Gap |
|---|---|---|---|
| HotpotQA | 8B | comparison | -9.8 |
| HotpotQA | 70B | comparison | -4.9 |
| 2WikiMHQA | 8B | comparison | -17.5 |
| 2WikiMHQA | 70B | comparison | -2.5 |
| 2WikiMHQA | 8B | inference | -25.9 |
| 2WikiMHQA | 70B | inference | -3.7 |

The rule's "comparison/inference → Generic" branch is robust across both model sizes and on every dataset where these types exist. This is the most consistent finding in the analysis.

### Implication for the routing rule

- **The "comparison/inference → Generic" branch is robust** — holds across 8B and 70B, on every dataset where the type exists.
- **The "bridge → SPARQL" branch only works on 70B.** On 8B (without GW), Generic wins on bridge questions on HotpotQA and 2WikiMHQA. The branch fails on 8B in the no-GW regime.
- **The model-size effect dominates the type effect**: which method wins is more about 8B-vs-70B than about question type.

So the routing rule's per-type premise is **half-validated**: the comparison/inference branch holds universally; the bridge branch only holds on 70B in the no-GW regime.

This connects back to the paper's deployed configuration: **GW is what rescues SPARQL+8B on bridge questions** (per Table 6), pulling the bridge branch back into the lead. Without GW, Generic alone would be the better default on 8B for nearly all question types.
