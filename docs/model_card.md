# Model card and limitations

## What the system is

A retrieval and verification pipeline over SEC 10-K filings. By default it uses
**no trained model**: parsing is rule-based, retrieval is BM25 + TF-IDF,
answers are extractive, and verification is symbolic. Setting
`ANTHROPIC_API_KEY` swaps the extractive generator for Claude
(`claude-opus-5-5` by default, override with `FILINGS_LLM_MODEL`), which is told
to answer only from the numbered evidence and to cite every sentence. The
verifier then checks that output exactly as it checks the extractive one.

## Intended use

* Analysts and students exploring what a company disclosed and how its risk
  disclosures changed year over year.
* A reference implementation of "citations + automatic verification" for
  financial question answering.

Not intended for investment advice, or for any decision that does not include a
human reading the cited source.

## Measured performance (from `eval/results/results.md`)

| Component | Test | Result |
|---|---|---|
| Retrieval (hybrid) | 36 hand-written questions, gold sentence regex | Hit@1 0.806 · Hit@5 0.917 · MRR 0.847 |
| Retrieval (BM25 only) | same | Hit@1 0.806 · Hit@5 0.944 · MRR 0.863 |
| Scoping | same | 100% of questions mapped to the right filing |
| Verifier | 1,072 synthetic perturbations of real sentences | P 1.000 · R 0.994 · F1 0.997 |
| Verifier | 30 hand-written paraphrases and subtle errors | accuracy 0.533 · F1 0.588 |
| Numeric QA | 171 metric questions | iXBRL route 1.000 vs. retrieval-only 0.035 |

## Known limitations

1. **The verifier is weak on paraphrase.** On the hand-written set, 9 of 15
   faithful paraphrases were flagged (false positives) because their wording
   differs from the source, and it misses errors that need world knowledge or
   logic: "possibility stated as fact", overgeneralisation ("all of its
   software"), attributes swapped between two entities in the same sentence,
   and number words ("five markets"). The synthetic score is high because those
   perturbations target exactly what the rules check, so read it as a unit
   test, not as a generalisation estimate. An NLI cross-encoder or an LLM judge
   is the natural next stage (see roadmap).
2. **Hybrid retrieval does not beat BM25 on this set.** The difference is one or
   two questions out of 36, so the hybrid default was kept rather than tuned
   to the test set. There is no dense embedding model, so purely semantic
   questions with no shared vocabulary will miss.
3. **Small corpus.** Nine filings from six large US issuers (three
   tech/AI, one automaker, two banks). Bank 10-Ks (Citi, JPMorgan) put MD&A in
   an exhibit, so their narrative coverage is mostly Item 1A.
4. **Risk-factor splitting relies on typography.** Issuers that do not bold or
   italicise risk headlines will produce coarser factors.
5. **Theme intensity is keyword-based** (13 hand-written regex lexicons). It
   shows direction of change well; absolute levels depend on the lexicon.
6. **The synthetic eval was designed alongside the verifier; the hand-written
   hard set was written afterwards and was not used for tuning.**

## Data

Nine public 10-K filings (see `data/manifest.json`). SEC filings are public
records. The HTML used to build this corpus came from the test fixtures of the
MIT-licensed [edgartools](https://github.com/dgunning/edgartools) project,
which are verbatim EDGAR documents; `make data` downloads the same filings
directly from SEC EDGAR instead.
