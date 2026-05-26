# SPARQL syntax-validity rate (eQxk #6, w3DN W2)

Direct quantification of the syntax-overhead hypothesis (paper Section 5.4):
8B vs 70B parse rate of the SPARQL block in SPARQL CoT outputs, on the
no-GW CSVs (500 questions each, paper-era timestamps per
[data_inventory.md](data_inventory.md)).

## Method

For each row in the 6 SPARQL CoT CSVs:
1. Locate the first `SELECT ... { ... }` block in the `sparql_cot` field
   (extract via brace-balancing).
2. Wrap bare predicate names (e.g. `name`, `releasedBy`) in `<...>` so they
   parse as IRIs. The model uses bare names by design (no schema is given
   to it), so without this normalization basically nothing parses; this
   step lets us measure *structural grammar validity* (well-formed
   SELECT/WHERE/triple-pattern syntax), not vocabulary compliance.
3. Try `rdflib.plugins.sparql.parser.parseQuery`. Count as parsed if it
   succeeds.

Script: [sparql_syntax_validity.py](sparql_syntax_validity.py). No new
LLM calls.

## Results

| Dataset | Model | N | Extract% | Parse-of-extracted% | Parse-of-all% |
|---|---|---|---|---|---|
| HotpotQA | 8B | 500 | 96.2 | 85.0 | **81.8** |
| HotpotQA | 70B | 500 | 99.4 | 93.2 | **92.6** |
| 2WikiMHQA | 8B | 500 | 98.2 | 93.1 | **91.4** |
| 2WikiMHQA | 70B | 500 | 96.4 | 99.0 | **95.4** |
| MuSiQue | 8B | 500 | 96.2 | 92.5 | **89.0** |
| MuSiQue | 70B | 500 | 96.6 | 95.4 | **92.2** |

**Per-model averages (parse-of-all):** 8B 87.4%, 70B 93.4%, gap 6.0 pp.

## Read

Both models attempt a SPARQL query on essentially every question (extract
rate 96-99%), so the difference is in the cleanness of what gets produced,
not in whether the model engages with the format.

8B's parse rate is 6 pp lower on average. The largest gap is on HotpotQA
(10.8 pp), which is also where 8B's SPARQL CoT deficit relative to Generic
CoT is largest (Table 5 / extended Table 5: SPARQL 70.6 vs Generic 75.6 on
8B, gap -5 pp).

This is direct support for the syntax-overhead hypothesis: 8B writes more
malformed SPARQL, and the dataset where it writes the most malformed
SPARQL is where the SPARQL-vs-Generic gap is largest. Not a complete
account of why SPARQL hurts 8B (alternative explanations like reasoning-
trace-length or post-training distribution alignment remain), but it
converts one mechanism from hypothesis to demonstrated.

## Caveats

- Measures *structural grammar* (well-formed SELECT/WHERE/triple patterns),
  not vocabulary correctness. A query with the right shape but a wrong
  predicate name still passes; a query with a typo'd brace fails.
- The bare-predicate-wrap normalization is necessary because the model is
  not given a schema; it uses NL predicate names. Without normalization
  parse rates would be near zero for both models and the comparison would
  be uninformative.
- Some rows have multiple SELECT blocks (the model wrote a draft and a
  revision); we extract the first one. A more lenient "any SPARQL block
  parses" would give slightly higher rates but the 8B/70B gap is robust.
