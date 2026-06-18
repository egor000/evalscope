# Handout B — Why This Matters and How to Use It

*For developers, test engineers, product, and the customer team.*

## What this changes for the customer conversation

Today, answering "is this model good enough for your workload?" means running the full benchmark suite on every candidate — slow and expensive enough that we hesitate to do it per-customer. These pruners change the unit of work from "measure the model's exact score" to "answer the customer's yes/no question and stop." A model that's clearly above or clearly below the customer's bar gets resolved in a small fraction of the samples; only genuinely borderline models cost the full run. In practice that turns a go/no-go from a batch job into something you can run live against a customer's specific bar, on multiple candidate models, in the time you used to spend on one.

Crucially, the output is a **decision with a confidence guarantee**, not a number someone has to interpret. The verdict is three-way — **PASS** (clears the bar), **FAIL** (below it), **BORDERLINE** (too close to call at this budget) — and "borderline" is an honest answer, not a failure. It tells the customer team exactly when a capability needs the full benchmark before we make a promise.

## How to run it tomorrow

A sales engineer or deployment lead does not need to touch eval code. Point evalscope at the candidate model, name the benchmark for the capability in question (coding → LiveCodeBench, long-context → AA-LCR), and set one number: the customer's accuracy bar.

```bash
DATASET_ARGS='{
  "live_code_bench": {
    "early_stop": {"target_accuracy": 0.60, "prune_ratio": 0.3, "risk_assessment": "balance"}
  }
}'

evalscope eval --model <candidate> --datasets live_code_bench \
  --dataset-args "$DATASET_ARGS" --output ./results/
```

`target_accuracy` is the customer's bar. `risk_assessment` picks how cautious to be (`conservative` when a wrong "yes" is costly). The run prints **PASS / FAIL / BORDERLINE** and how many samples it actually needed. That's the whole workflow — change the model, change the bar, re-run.

## What the multimodal probe gives that random sampling cannot

Random sampling tells you an *average* score across a mixed bag of images. That average hides the failure that actually matters: a model can look fine on easy pictures while its image encoder quietly falls apart on the hard ones — dense charts, small text inside an image, fine diagrams. Those are exactly the images many real customer workloads depend on. The probe is built to *seek out* that failure mode rather than average it away: it stresses the encoder with demanding images and compares performance when detail is degraded. The result isn't "this model scores X on images" — it's "this model's vision holds up / breaks down on the kind of images your product actually uses." Random sampling cannot give that answer because it isn't looking for it.

## Why a customer-facing PM should care

Two reasons, both about trust. First, **speed-to-answer**: you can give a customer a defensible go/no-go on their exact bar quickly enough to keep a deal moving, and re-run it the moment they revise the bar or a new model lands. Second, **honesty as a feature**: the method refuses to over-claim. When it says BORDERLINE, that's a signal we've earned the right to send — it protects the customer from a model that's not ready and protects us from a promise we can't keep. A PM can stand behind a PASS because the same procedure that produces it is the one that would have flagged a FAIL.
