# Laya decisions

`LayaDecisionOperation` owns local tokenizer loading, bounded question batching,
calibration and typed results. `MereRunLayaModel` owns the native neural network.
CLI, API and preflight share request validation and sequence construction.

Requests use ordered question and criterion arrays. State is UTF-8 text; callers
can serialize structured state into that string. Choice criteria accept strings
or `{ "label": "...", "description": "..." }` objects. Score criteria appear
in ascending ordinal order. Noul evaluates false/true and can describe either
label. The action head reports the model's probability of acting; it executes
no external action.

Only the selected checkpoint subdirectory is downloaded. Managed checkpoints
pin the Hugging Face revision independently of the reference SDK revision.
Local checkpoint paths name the directory containing `rl_agent_config.json`.
Preflight loads config/tokenizer files and reports token-budget truncation;
weight shape checks occur when the native network is loaded.
