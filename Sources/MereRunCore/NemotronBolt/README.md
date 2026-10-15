# Nemotron 3.5 Bolt orchestration

This directory owns managed checkpoint identity, tokenizer loading, image
preparation, chat/tool responses, and shared autoregressive decoding.
`MereRunNemotronBoltModel` owns all model computation and packed weight loading.

Run `swift test --filter NemotronBolt` and `./scripts/check.sh` after changes.
The 2026-10-15 checkpoint report under `docs/qualification/` records the
initial text/image execution, speed, and memory diagnostics. General model
quality and long-context memory remain separate qualification gates.
