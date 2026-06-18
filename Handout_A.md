# Handout A — Why This Works

*For an engineer who could have built this. Each section answers one required question directly.*

## 1. What problem I understood myself to be solving

Sales does not need the model's accuracy; they need a **decision**: is each capability good enough for this customer's bar? That reframes pruning from "estimate a number to ±ε" into "stop sampling the moment the go/no-go answer is statistically resolved." A sequential test is the optimal answer to that exact question — it spends samples only until the evidence crosses a decision boundary, so a model clearly above or below bar costs a handful of samples while only genuine borderline cases pay the full budget.

I run **one test per benchmark, never pooled.** The customer question is per-capability (coding good enough? long-context good enough?); pooling would let strength on LiveCodeBench mask weakness on AA-LCR. Each test is an opt-in `early_stop` block on the existing benchmark adapter, so the machinery composes with any future dataset.

Three design choices make this defensible rather than a dressed-up random sample:

- **Finite-population SPRT, not vanilla Wald.** N is small (315 / 100) and we sample *without replacement* from a fixed, fully-enumerated population. So hypotheses are cast as population success **counts** `k0 = round(N·p0)`, `k1 = round(N·p1)`, and the likelihood-ratio increments use the hypergeometric form. This curtails automatically — once the unseen remainder can no longer change the verdict, it stops — and degrades gracefully to "run the whole benchmark" at n=N with no truncation artifact.
- **An honest gray zone.** The indifference band `[p0, p1]` is derived from budget and risk profile via a finite-population normal margin. With only 100 AA-LCR samples, even the full benchmark resolves accuracy to ≈±10 points; the band advertises that limit instead of faking 5-point resolution.
- **Selection consumes zero model-score bits.** This is the answer to the unseen-fourth-model and forbidden-baseline requirements. The only structure injected is **proportional stratified ordering** over sample *features* (LiveCodeBench difficulty/platform/date; AA-LCR context length/domain) — never the shipped models' correctness. The three shipped models are used only to *simulate* the validation curves; they never touch selection.

For AA-LCR I also correct LLM-judge noise: I hand-graded all 300 (model × sample) predictions blind to the judge, built an FPR/FNR confusion matrix, and transform the hypotheses into observed space (`q = fpr + (1 − fnr − fpr)·p`), running SPRT on the observed verdicts against `q0, q1`. LiveCodeBench's sandbox grader is deterministic, so its noise channel is simply off.

## 2. How much I pruned, and why the subset is sufficient

Reported as **expected sample count (ASN)** from operating-characteristic simulations, with the fixed-budget Wilson `n` as the worst-case fallback — not a single fixed number, because the whole value of a sequential test is that confident models stop early. Clear-cut models resolve in tens of samples; only true borderline cases approach full budget. The subset is *sufficient by construction of the stopping rule*: the test only declares PASS/FAIL when the finite-population likelihood ratio (or aGRAPA confidence sequence) crosses a boundary calibrated to the configured Type-I/Type-II rates, so "we stopped" is identical to "the go/no-go answer is resolved at the stated confidence." When it cannot resolve within budget it returns BORDERLINE rather than guessing.

## 3. Part B — what I would do, and why it stresses image encoders *specifically*

If the customer adds multimodal next quarter, the question narrows to **is the image encoder good enough** — not generic VQA skill. The trap is that raw MMMU accuracy conflates *reasoning* failure with *perception* failure. A pruned set that just samples MMMU broadly measures the wrong thing. My probe isolates the encoder:

**Which images I select (and why each stresses the encoder):**
- **High spatial-frequency content** — dense scientific diagrams, plots, tables, sheet music, chemical structures. Encoders downsample to a fixed patch grid; fine detail is the first thing lost to limited resolution, so these expose encoder capacity directly.
- **Text-in-image / OCR-dependent items** — small or stylized text inside the figure. Reading it requires the encoder to preserve sub-patch detail; a weak encoder forces the model to guess.
- **Small, low-contrast, or cluttered targets** — the answer hinges on a visual detail occupying few pixels. This is precisely where encoder resolution and tokenization degrade, while a strong reasoner with a weak encoder cannot compensate.

**How I measure encoder quality through the standard OpenAI interface (no logits):** *paired contrasts*, where the signal is the **gap**, not the absolute score.
- *Image vs. text-transcription of the same content* — a real encoder should beat the text-only baseline; collapse to text-only performance means the vision path adds nothing.
- *Clean vs. perturbed image* (downscale, JPEG-compress, crop) — a robust encoder degrades gracefully; a weak one falls off a cliff. The slope of that fall is the encoder-quality measurement.

**Pruning strategy:** stratify the full ~12K by subject × image-type, select the perception-heavy strata and within them the paired-contrast items, and size the probe with the same sequential stopping rule against an encoder-quality bar. Stratifying on image *type* (not subject alone) is what makes it encoder-specific: it concentrates the budget where encoders fail, instead of averaging that failure into a subject-balanced score.

## 4. Assumptions about distribution, scale, and model behavior

- **Distribution:** strata are defined from metadata available *before* scoring; the sample set is a fixed, fully-enumerated population (sampling without replacement). Proportional ordering keeps each draw marginally population-distributed, so the pooled finite-population statistic stays valid. The hypergeometric likelihood is exact under random order and *conservative* under proportional order (verified by permutation simulation).
- **Scale:** N is small enough that finite-population correction matters and that more than ~3–5 strata makes proportional interleaving meaningless (a 10% stratum of 100 is 10 items). The method is designed for the hundreds-of-samples regime, not asymptotics.
- **Model behavior:** per-sample scores are accuracy-like in [0,1]. The judge confusion matrix is assumed to **transfer to the fourth (unseen) model** — checked via per-model stability of the matrix; if it diverges, fall back to conservative bounds on FPR/FNR. No assumption is made that the fourth model resembles the three shipped ones in accuracy — that is exactly why selection uses no model scores.

## 5. What would change with…

**(a) More data.** AA-LCR's 100-sample ceiling is the binding constraint — even the full benchmark only resolves ≈±10 points. More samples narrow the achievable gray zone directly, and would let me switch from proportional to **Neyman allocation** (oversampling high-variance strata) for a real variance discount, which the small-N regime currently can't support safely.

**(b) A live model endpoint.** Two upgrades unlock. First, repeated judging on a subsample would **decompose** judge error into systematic bias vs. stochastic noise (hand-calibration only gives the combined marginal rate). Second, the Part B probe becomes *executable* rather than design-only: the image-vs-transcription and clean-vs-perturbed paired contrasts require live queries, so with an endpoint I run them directly instead of proposing them.

**(c) More time.** Integrate FPR/FNR uncertainty out as Beta posteriors (turning the screen into a sequential Bayes-factor test); add the weighted group-sequential estimator to harvest the stratification variance the pooled statistic currently leaves on the table; and expand leave-one-model-out validation into a full synthetic-model sweep across shifted accuracies to map the entire operating-characteristic surface.

