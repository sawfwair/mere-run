# vendor/ds4

Prebuilt DeepSeek V4 Flash inference binaries vendored from
[ds4](https://github.com/antirez/ds4.git) at commit `f8b4ed635d559b3a5b44bf2df6a77e21b3e9178f`.

Rebuild with:

    scripts/rebuild_ds4.sh

Binaries:
- `ds4`         interactive CLI (not used by mere.run runtime; included for parity)
- `ds4-server`  OpenAI-compatible HTTP server (spawned by MereRunCore)
- `ds4-bench`  frontier throughput benchmark

The 86 GB GGUF model is **not** vendored. mere.run lazy-downloads it from
`antirez/deepseek-v4-gguf` on Hugging Face the first time the premier agent
tier is used on a 96 GB+ Apple Silicon Mac.
