# ZImageI2L

Image-to-latent support for ZImage workflows.

- `ZImageI2LConfigs.swift`: typed configuration.
- `ZImageI2LResources.swift`: resource discovery.
- `ZImageI2LGenerator.swift`: runtime orchestration.
- `Model/`: native encoder/model components.

Keep model-specific tensor preparation here and public command behavior in the
CLI layer.

Supported attention shapes use MLX fused SDPA by default. Set
`MERERUN_FUSED_SDPA=0` to restore materialized attention for diagnosis.
