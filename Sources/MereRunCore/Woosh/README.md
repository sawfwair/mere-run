# Woosh Runtime

Swift-native MLX runtime for Woosh sound-effect generation.

## Managed model

- Canonical ids: `sfx-woosh-dflow`, `sfx-woosh-flow`,
  `sfx-woosh-clap`, `sfx-woosh-synchformer`, `sfx-woosh-dvflow-8s`,
  `sfx-woosh-vflow-8s`
- Upstream: `SonyResearch/Woosh@v1.0.0`
- Mirror used by managed installs: `AEmotionStudio/woosh-models`
- Serving engine: `woosh`

The managed install is a structured root with:

- `checkpoints/Woosh-DFlow/`, `checkpoints/Woosh-Flow/`,
  `checkpoints/Woosh-DVFlow-8s/`, `checkpoints/Woosh-VFlow-8s/`, or
  `checkpoints/Woosh-CLAP/`
- `checkpoints/Woosh-AE/`
- `checkpoints/TextConditionerA/` or `checkpoints/TextConditionerV/`
- `checkpoints/TextConditionerA/tokenizer/`,
  `checkpoints/TextConditionerV/tokenizer/`, or
  `checkpoints/Woosh-CLAP/tokenizer/`

The raw-video V2A path also uses the `sfx-woosh-synchformer` companion model,
which stores `mmaudio_synchformer_fp16.safetensors` from
`Kijai/MMAudio_safetensors`.

The tokenizer folder is mounted from `FacebookAI/roberta-large` because the
public Woosh release ships the text-conditioner weights separately from the
RoBERTa tokenizer assets.

## Architecture

- `WooshResources.swift`: model ids, resource layout, validation, sampler
  defaults, and user-facing errors.
- `WooshRobertaTextEncoder.swift`: RoBERTa-large style text conditioner and
  tokenizer loading.
- `WooshDiT.swift`: native Flow and FlowMap DiT blocks used by
  `Woosh-Flow` and `Woosh-DFlow`.
- `WooshVideoDiT.swift`: VFlow/DVFlow video-feature modality blocks and
  samplers.
- `WooshCLAP.swift`: Woosh-CLAP text/audio scoring with RoBERTa and PaSST.
- `WooshSynchformer.swift`: native MotionFormer/Synchformer visual extractor
  for raw-video V2A conditioning.
- `WooshVocosDecoder.swift`: Woosh-AE latent encoder/decoder and Vocos-style
  decoder.
- `WooshGenerator.swift`: Flow/DFlow/VFlow/DVFlow sampling loops,
  classifier-free guidance, latent decode, and PCM output.

## Scope

The public CLI surface is `mere.run sfx`, not `music`, because Woosh is for
sound effects. The implemented public paths are text-to-SFX through
`Woosh-DFlow` and `Woosh-Flow`, Woosh-AE encode/decode utilities,
Woosh-CLAP scoring, and VFlow/DVFlow generation from raw video or precomputed
Synchformer `synch_out` tensors.

Keep stdout machine-readable in CLI callers. Progress and diagnostics should go
to stderr.
