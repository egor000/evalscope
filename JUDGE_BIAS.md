# Judge Bias Calibration

This note records a small calibration pass on `aa_lcr` using three model result
files. The goal was to estimate whether the LLM judge systematically marks
correct answers as incorrect or incorrect answers as correct.

## Data

Reviewed 300 judged model/sample pairs:

- `aa_lcr__gpt-oss-120b`: 100 samples
- `aa_lcr__kimi-k2.5`: 100 samples
- `aa_lcr__minimax-m2.5`: 100 samples

Each model answer was compared against the official AA-LCR `answer` field and
the stored judge verdict. I kept only clear cases where the judge verdict looked
wrong; borderline numeric rounding, format-only differences, or answers that
mentioned the reference while adding wrong extra content were not counted.

## Findings

Curated likely judge mistakes:

| Model | Judge false positives | Judge false negatives |
| --- | ---: | ---: |
| `gpt-oss-120b` | 0 | 1 |
| `kimi-k2.5` | 0 | 3 |
| `minimax-m2.5` | 0 | 0 |
| **Total** | **0** | **4** |

Confusion matrix against the curated assistant review:

| Model | Judge correct / assistant correct | Judge correct / assistant incorrect | Judge incorrect / assistant correct | Judge incorrect / assistant incorrect |
| --- | ---: | ---: | ---: | ---: |
| `gpt-oss-120b` | 48 | 0 | 1 | 51 |
| `kimi-k2.5` | 66 | 0 | 3 | 31 |
| `minimax-m2.5` | 64 | 0 | 0 | 36 |

## Rate Calculation

For the judge-bias correction model:

```text
q = fpr + (1 - fnr - fpr) * p
```

use class-conditional rates:

```text
fnr = false negatives / assistant-correct cases
fpr = false positives / assistant-incorrect cases
```

From the curated matrix:

```text
assistant-correct = 48 + 1 + 66 + 3 + 64 + 0 = 182
assistant-incorrect = 0 + 51 + 0 + 31 + 0 + 36 = 118

fnr = 4 / 182 = 0.0220
fpr = 0 / 118 = 0.0000
```

Recommended early-stop config:

```json
{
  "judge_false_negative_rate": 0.022,
  "judge_false_positive_rate": 0.0
}
```

