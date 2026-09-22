# Text Decide

## Purpose

Evaluate ordered choice, score, and boolean questions with native Laya.

## Usage

Create a JSON request with `state` text and a `questions` array; each question
has `id`, `type`, `instructions`, and, for choice or score, ordered `criteria`.

```bash
mere.run model pull text-decide-laya
mere.run text decide --model text-decide-laya --input request.json --preflight --pretty
mere.run text decide --model text-decide-laya --input request.json --output decisions.json --pretty
mere.run guide --model text-decide-laya
```

Use the model handbook for a complete request example, checkpoint selection,
API behavior, token budgets, and calibration limits. Output is always JSON.

## Sources

- [Laya model repository](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982)
- [Laya SDK reference](https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af)
