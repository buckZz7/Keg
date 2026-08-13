# Multilingual gate decision + corpus quality (OPEN — resumed next session)

Status: **decision in progress, paused for a break.** This doc is the handoff.

## The core question

Our fidelity gate is per-component: worst-component mean-KL ≤ 0.02 (ACCEPT_KL),
and **multilingual is always the worst component** (it is the binding constraint
at every quant we measured). Two related decisions are entangled:

1. **Is the multilingual *content* in our corpus bad?** (we found it partly is)
2. **Should multilingual be a hard per-component *gate* at all?** (field says no)

## Evidence gathered (this session)

### The multilingual floor is a real, large effect
Per-component worst-KL across measured quants (from `receipts/`):

| Quant | prose | code | technical | **multilingual** |
|---|---|---|---|---|
| Q8_0 (29.6 GB) | 0.0049 | 0.0016 | 0.0014 | **0.0151** |
| UD-Q6_K_XL (26.3 GB) | 0.0099 | 0.0030 | 0.0020 | **0.0165** |
| Q6_K crown (22.9 GB) | 0.0190 | 0.0052 | 0.0039 | **0.0199** |

Multilingual is the max component at every quant AND has a high **floor even
near-lossless** (0.0151 at Q8_0, while prose is 0.0049). It barely moves with
compression (0.0151→0.0165→0.0199) while prose moves 5× more. This is because
multilingual is a high-entropy distribution where the model is genuinely less
confident — so identical weight perturbations produce ~3× more KL. The floor
consumes ~75% of the entire 0.02 gate budget before a recipe even competes.

### Corpus artifact found: disambiguation pages
~14% of the multilingual component (16 of 111 docs) were Wikipedia
**disambiguation pages** — list-of-meanings pages (album titles, band names,
multi-script) that no model can predict, inflating the multilingual KL floor.
Examples: "Music kan verwijzen naar: Music (album van Madonna)...", "Sun steht
für: Sun (Familienname)...". Triggered by ambiguous titles (Sun, Music, Human,
Earth, History, Philosophy, Language, Water) hitting disambig pages on many wikis.

### Field standard does NOT gate multilingual
- **Unsloth** calibration (Calibration_v3/v5) is **English-centric** — v5 is
  community-noted as "lacking major languages such as French," ~15× smaller than
  a full multilingual set. Unsloth produced the crown's vendor models.
- **bartowski** includes multilingual as ONE component of a diverse mix (papers,
  code, dialogues, math, multilingual) — but not as a hard gate.
- Neither treats multilingual as a strict per-component acceptance threshold.
  Our setup is **stricter than the tools miners actually use** — a quant that
  passes perfectly on Unsloth/bartowski's own standards could fail Keg purely on
  the multilingual component.

## The two fixes identified

### Fix 1 — Corpus quality (DONE in code, NOT yet rebuilt)
`tools/build_corpus.py` updated:
- **`is_disambiguation()`** — language-family disambiguation markers + structural
  heuristic (parenthetical-heavy / short / title-case lists). Unit-tested in
  `run_tests.py` (`test_disambiguation_filter`).
- **Switched to langlinks concept-resolution** — resolve each concept (Physics,
  Chemistry, ...) to its canonical title in every language via Wikipedia's
  interlanguage links, so we fetch the *concept*, not a same-named band/album.
  (Batched multi-title langlinks fails — must query per-concept; verified 198/220
  concept-lang pairs resolve.)

⚠️ **NOT DONE:** the corpus was NOT rebuilt with the fixed builder. We restored
`corpus/production.txt` to the last-good committed version (111 multilingual
docs, still contains the disambiguation junk). Rebuilding needs a ~10-min network
run (`python3 tools/build_corpus.py`) + re-deriving the reference on GPU.

### Fix 2 — Gate policy (OPEN DECISION, not decided)
Spectrum for how multilingual should be gated:
- **A. Keep hard per-component gate (current)** — strictest, stricter than field.
- **B. Multilingual in aggregate, no hard gate** — matches Unsloth/bartowski.
  Gate keys on prose/code/technical (gated) + multilingual folds into the mean.
- **C. Middle** — keep a multilingual gate but at a *looser* bound (it's a harder
  distribution), not the same 0.02 as English components.

Buck leaned toward **B-with-nuance**: gate per-component on prose/code/technical
(preserves anti-overfit design — smcleod's "falls apart on code" concern), treat
multilingual as measured-and-reported-in-the-mean, not a gate. Not finalized.

## Recommended resume order
1. Decide **A vs B vs C** for multilingual gating.
2. Rebuild corpus with the fixed builder (Fix 1) — needs network (~10 min).
3. Re-derive the reference on GPU (A100) against the clean corpus → see
   multilingual's REAL floor once disambiguation junk is gone (it may drop
   enough that the "floor problem" partially self-resolves and B matters less).
4. Then the format decision (GGUF vs format-open) — the format-open roadmap is
   currently parked; GGUF-only is the working recommendation, but the format
   question was deferred pending the multilingual decision.

## Notes
- Code committed this session: `tools/build_corpus.py` (disamb filter + langlinks
  concept resolution, tested), `tools/run_tests.py` (new disamb test).
- Corpus files left at last-good committed state; reference.json / receipts
  unchanged (still reflect the old corpus).
