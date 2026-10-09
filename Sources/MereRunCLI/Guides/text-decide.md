# Text Decide

## Purpose

Evaluate choice, score, and boolean questions with native Laya or Clef.

## Usage

For Laya, create a JSON request with `state` text and a `questions` array; each question
has `id`, `type`, `instructions`, and, for choice or score, ordered `criteria`.

```bash
mere.run model pull text-decide-laya
mere.run text decide --model text-decide-laya --input request.json --preflight --pretty
mere.run text decide --model text-decide-laya --input request.json --output decisions.json --pretty
mere.run guide --model text-decide-laya
```

Use the model handbook for a complete request example, checkpoint selection,
API behavior, token budgets, and calibration limits. Output is always JSON.

For `text-decide-clef-4bit` or `text-decide-clef-flash-4bit`, `state` may be text or structured JSON and `questions`
is an object keyed by question ID. Choice criteria are an object of option IDs
to descriptions; score criteria are an ordered array. Clef also accepts local
image paths and video frame-path arrays, and evaluates all fields jointly.

```bash
mere.run model pull text-decide-clef-4bit
mere.run text decide --model text-decide-clef-4bit --input clef-request.json --preflight --pretty
mere.run guide --model text-decide-clef-4bit
mere.run model pull text-decide-clef-flash-4bit
mere.run text decide --model text-decide-clef-flash-4bit --input clef-request.json --pretty
```

## Sources

- [Laya model repository](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982)
- [Laya SDK reference](https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af)
- [Clef MLX checkpoint](https://huggingface.co/mlx-community/clef-4bit/tree/e0a23bd4406c15075b7473616429c46f3fd130a9)

Clef Omni (`text-decide-clef-omni`) adds mixed local audio, image, and video
inputs to Clef's question object, with a 64,000-token context. Its original BF16
30B-A3B checkpoint recommends 96 GB or more unified memory. See
`mere.run guide handbook-clef` for limits and checkpoint qualification status.
