# vendor/ds4

Prebuilt DwarfStar inference binaries for DeepSeek V4 Flash vendored from
[antirez/ds4](https://github.com/antirez/ds4.git) at commit `b6af0adf8ca97c89145c9f9c15be70c9fd6c4507`.

Rebuild with:

    scripts/rebuild_ds4.sh

Release builds can preserve the hardened-runtime Developer ID signature with:

    DS4_CODESIGN_IDENTITY=<certificate-fingerprint> scripts/rebuild_ds4.sh

Binaries:
- `ds4`         interactive CLI (not used by mere.run runtime; included for parity)
- `ds4-server`  OpenAI-compatible HTTP server (spawned by MereRunCore)
- `ds4-bench`  frontier throughput benchmark

The DeepSeek V4 Flash 0731 Q2 imatrix GGUF (86.72 GB / 80.76 GiB) is **not**
vendored. mere.run lazy-downloads it from
`antirez/deepseek-v4-gguf` on Hugging Face the first time the premier agent
tier is used on a 96 GB+ Apple Silicon Mac.

The binaries include Iris image decoders. Their MIT notice is in
`IRIS-LICENSE`. Model support exposed by mere.run remains DeepSeek V4 Flash.
