# Qwen Image 2.1 Turbo metadata fixtures

Pinned upstream configuration files and safetensors tensor-header schemas only;
no trained-weight payloads. `source.json` records the repository revision and
source hashes. The upstream Qwen Research License and required attribution are
included in `LICENSE` and `Notice`.

Regenerate with `python3 scripts/validation/qwen-image-21-turbo-metadata.py`.
The tool uses bounded HTTP range reads and refuses full checkpoint downloads.
These fixtures validate metadata and native tensor contracts; they do not qualify
trained-checkpoint image quality.
