# Self-Ask (Press et al. 2022) Baseline

Added to address eQxk #1 ("the only prompting baselines are no-CoT and a 'generic CoT' that the authors wrote themselves").

## Methodology

- **Prompts**: verbatim from Press et al. 2022 Tables 13 (4-shot) and 10 (6-shot).
- **Adaptation**: a labeled `Context: {KET-RAG retrieval}` block inserted between demonstrations and test question (Press et al. use parametric knowledge or a search engine; we substitute retrieved context).
- **Prompt ending**: `Are follow up questions needed here: Yes.\nFollow up:` per Press et al. footnote 4 (recommended for smaller-than-Davinci models).
- **Decoding**: temperature 0.3, max_tokens 2048.
- **Symmetric comparison (headline)**: Self-Ask uses Press et al.'s 2WikiMHQA 4-shot prompt across all three test datasets (matching their Bamboogle precedent of "no prompt tuning"); SPARQL CoT uses its generic single-example prompt uniformly. Neither is tuned to the test dataset.

## Headline result

500 questions per dataset, both methods using non-test-dataset-tuned prompts and the same full retrieved context (no GW compression on either side, for apples-to-apples):

| Dataset | Model | Self-Ask | SPARQL CoT (ours) | gap |
|---|---|---|---|---|
| HotpotQA | 8B | 57.0% | **70.6%** | +13.6 pp |
| HotpotQA | 70B | 70.0% | **80.2%** | +10.2 pp |
| 2WikiMHQA | 8B | 42.2% | **45.6%** | +3.4 pp |
| 2WikiMHQA | 70B | 58.6% | **61.0%** | +2.4 pp |
| MuSiQue | 8B | 25.0% | **28.8%** | +3.8 pp |
| MuSiQue | 70B | **46.8%** | 43.8% | -3.0 pp |

**SPARQL CoT wins 5 of 6 cells.** The only exception is MuSiQue 70B (-3.0 pp, within typical 500-question noise).

## Caveat to disclose

When Self-Ask uses Press et al.'s MuSiQue-tuned 6-shot prompt (the published form), it wins MuSiQue on both models (43.4% / 47.4%). The MuSiQue prompt was tuned by Press et al. on the MuSiQue training set per Section 3.5; ours wasn't. The headline table uses the symmetric (non-test-tuned) comparison.

## Cost: not a concern

Mean output tokens per question — Self-Ask is consistently *shorter*, not longer, than SPARQL CoT (1.2-2.4× fewer tokens across cells). The cost-asymmetry concern doesn't materialize.

## Why Self-Ask specifically (eQxk named three: IRCoT, Self-Ask, Decomposed Prompting)

| Method | Native form | Single-call viability |
|---|---|---|
| IRCoT (Trivedi et al. 2023) | Multi-call (interleaved retrieval + generation per step) | Would require our own single-call adaptation |
| Decomposed Prompting (Khot et al. 2023) | Multi-call (separate decomposer + sub-task prompts) | Would require our own single-call adaptation |
| Self-Ask (Press et al. 2022) | Single-call by default (vanilla, no search engine) | Published as single-call with exact prompts |

Our setting is single-call methods on a fixed retrieved context. Self-Ask is the only one of the three whose vanilla form already fits — published verbatim prompts (Press et al. Tables 10/13), validated on 2WikiMHQA and MuSiQue, with documented prompt-transfer methodology (Bamboogle precedent) that enabled our symmetric comparison.

The other two require *us* to invent a single-call adaptation, which re-exposes the very "homemade prompt" critique that eQxk's #1 leveled at our generic CoT. Self-Ask reproduces a published method verbatim with zero prompt engineering on our side, addressing the critique most cleanly. A reviewer pushing for IRCoT or Decomposed Prompting can fairly be asked: which published single-call form?

## The fuller methodological defense (paper-ready paragraph)

> Of the three established multi-hop methods reviewer eQxk named (IRCoT, Self-Ask, Decomposed Prompting), only Self-Ask has a published single-call form. The other two are defined by their multi-call interactions: IRCoT by interleaved retrieval, Decomposed Prompting by sub-task handler delegation. A single-call adaptation of either strips out the contribution that distinguishes it from chain-of-thought. The single-call kernel of IRCoT collapses to chain-of-thought over the up-front context (which we test via generic CoT); the single-call kernel of Decomposed Prompting collapses to Self-Ask's decompose-and-answer-in-one-pass pattern (which we test directly). We therefore selected Self-Ask as the established single-call baseline, and would have re-exposed the homemade-prompt critique by inventing single-call adaptations of the other two.
