# vendor/ds4

Prebuilt DwarfStar inference binaries for DeepSeek V4 Flash vendored from
[antirez/ds4](https://github.com/antirez/ds4.git) at commit `4893e0c40fba03dbc85555faeb035799aa04e0b6`.

Rebuild with:

    scripts/rebuild_ds4.sh

Release builds can preserve the hardened-runtime Developer ID signature with:

    DS4_CODESIGN_IDENTITY=<certificate-fingerprint> scripts/rebuild_ds4.sh

Binaries:
- `ds4`         interactive CLI (not used by mere.run runtime; included for parity)
- `ds4-server`  OpenAI-compatible HTTP server (spawned by MereRunCore)
- `ds4-bench`  frontier throughput benchmark

The 86 GB GGUF model is **not** vendored. mere.run lazy-downloads it from
`antirez/deepseek-v4-gguf` on Hugging Face the first time the premier agent
tier is used on a 96 GB+ Apple Silicon Mac.
