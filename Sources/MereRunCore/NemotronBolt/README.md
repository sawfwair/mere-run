# Nemotron 3.5 Bolt orchestration

This directory owns managed checkpoint identity, tokenizer loading, image
preparation, chat/tool responses, and shared autoregressive decoding.
`MereRunNemotronBoltModel` owns all model computation and packed weight loading.

Run `swift test --filter NemotronBolt` and `./scripts/check.sh` after changes.
Released-checkpoint quality and memory remain separate qualification gates.
