# Architecture reading map

Start at the CLI surface, and then follow one runtime family at a time. This map
gives contributors a reliable reading order without explaining every model
detail.

For the broader documentation set, start at the
[mere.run documentation home](/).

## Start here

- CLI entry point: `Sources/MereRunCLI/MereRunCLI.swift`
- Shared CLI helpers: `Sources/MereRunCLI/Support`
- [Inference admission](./internals/inference-admission.md): `Sources/MereRunAdmission`
- [Runtime residency](./internals/runtime-residency.md): `Sources/MereRunResidency`
- Model paths and manifests:
  - `Sources/MereRunModelKit/MereRunModelPaths.swift`
  - `Sources/MereRunModelKit/MereRunModelManifest.swift`
  - `Sources/MereRunCore/ModelResolver.swift`

If you want to understand what a command does end to end, start at the command
file, and then use the table to open the family entry point.

## Durable operation history

`MereRunExecution` owns run-directory leases, file fingerprints, atomic record
writes, and terminal states. Core owns image records; AudioCore owns file
transcription records. CLI inspection and listing use one typed operation-record
adapter. See [shared transcription](./internals/speech-transcription-operation.md)
and [shared image generation](./internals/image-generation-operation.md) for
preparation, recovery, and retry responsibilities.

## Shared chat execution

Read [shared chat execution](./internals/chat-execution.md) for request resolution,
native runtime selection, and cleanup. Core owns the common request and
generation path. CLI and API adapters own presentation, tool authorization,
wire compatibility, and admission or model-residency scope.

## Shared text training

Read [shared text training](./internals/text-training-execution.md) for typed
options, dataset preparation, native pipeline dispatch, and manifest publication.
The CLI owns presentation and the optional dashboard; native trainers retain
optimizer and checkpoint behavior.

## Shared autoregressive decoding

Read [shared decode boundaries](./internals/decode-runtime-boundaries.md) before
changing a family decoder. `Sources/MereRunDecode` owns sampling, pipelined
loops, streaming, and logprob capture. `Sources/MereRunKVCache` owns attention
cache implementations, including optional affine quantization. Model runtimes
supply forward callbacks and own prompt preparation and resource cleanup.

## Qwen text and vision-language models

Read [Qwen runtime boundaries](./internals/qwen-runtime-boundaries.md) for the
model library and generator stages. Start at `Sources/MereRunCore/Q35/Q35Generator.swift`
for requests, then follow loading, prefill, and decode extensions. Model math
lives in `Sources/MereRunQwenModel`.

## Gemma text and vision-language models

Read [Gemma runtime boundaries](./internals/gemma-runtime-boundaries.md) for the
model and generator stages. Start at `Sources/MereRunCore/Gemma4/Gemma4Generator.swift`
for requests, then follow its loading, prefill, and decode extensions. Model math
and cache implementations live in `Sources/MereRunGemmaModel`.

## H3 video and Laguna text models

Read [H3 and Laguna runtime boundaries](./internals/h3-laguna-runtime-boundaries.md)
for model ownership, generator stages, shared vocoder layers, and validation.
H3 computation lives in `Sources/MereRunH3Model`; Laguna computation lives in
`Sources/MereRunLagunaModel`. Their Core generator extensions own loading,
request execution, conditioning or batching, and cleanup.

## Image families

Read the [shared image operation](./internals/image-generation-operation.md)
and the [image model boundaries](./internals/image-runtime-boundaries.md)
before following a family implementation. CLI, API, and preflight adapters use
`ImageGenerationPlan` for resolution and validation, and execution passes the
resolved request to `ImageGenerationOperation`.

Klein image generation:

- CLI: `Sources/MereRunCLI/Commands/ImageGenerateCommand.swift`
- Runtime entry point: `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator.swift`
- Read next:
  - `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+ModelLoading.swift`
  - `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Generation.swift`
  - `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Denoising.swift`
  - `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Output.swift`

ZImage generation:

- Runtime entry point: `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator.swift`
- Read next:
  - `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+ModelLoading.swift`
  - `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+Inference.swift`
  - `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+Denoising.swift`
  - `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+Output.swift`

Qwen image editing:

- Runtime entry point: `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator.swift`
- Read next:
  - `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+ModelLoading.swift`
  - `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+Encoding.swift`

SenseNova U1.5 generation and editing:

- Runtime entry point: `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Generator.swift`
- Read next:
  - `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Model.swift`
  - `Sources/MereRunCore/SenseNovaU15/SenseNovaU15VisionAndHead.swift`
  - `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Tokenizer.swift`
  - `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Scheduler.swift`
  - `Sources/MereRunCore/SenseNovaU15/SenseNovaU15ImageIO.swift`
  - `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Resources.swift`

Krea 2 generation and LoRA training:

- Runtime entry point: `Sources/MereRunCore/Krea2/Krea2Generator.swift`
- Read next:
  - `Sources/MereRunCore/Krea2/Krea2RawResources.swift`
  - `Sources/MereRunCore/Krea2/Krea2Resources.swift`
  - `Sources/MereRunCore/Krea2/Krea2LoRAInjector.swift`
  - `Sources/MereRunCore/Krea2/Krea2LoRATrainer.swift`
  - `Sources/MereRunCore/Krea2/Krea2Configs.swift`
  - `Sources/MereRunCore/Krea2/Krea2Model.swift`
  - `Sources/MereRunCore/Krea2/Krea2ModelLoader.swift`
  - `Sources/MereRunCore/Krea2/Krea2SampleBuilder.swift`

LFM2.5 text generation and LoRA training:

- Runtime entry point: `Sources/MereRunCore/LFM2/LFM2Generator.swift`
- Read next:
  - `Sources/MereRunCore/LFM2/LFM2Model.swift`
  - `Sources/MereRunCore/LFM2/LFM2TextModelLoader.swift`
  - `Sources/MereRunCore/LFM2/LFM2TextLoRAInjector.swift`
  - `Sources/MereRunCore/LFM2/LFM2TextLoRATrainingPipeline.swift`

Shared text encoder stack used by image models:

- Public entry point:
  `Sources/MereRunTextEncoder/TextEncoder.swift`
- Architecture internals:
  - `Sources/MereRunTextEncoder/TextEncoder+RoPE.swift`
  - `Sources/MereRunTextEncoder/TextEncoder+Blocks.swift`

## Speech stack

Read [Speech runtime boundaries](./internals/speech-runtime-boundaries.md) for
the dependency map. Qwen ASR, Qwen TTS, Parakeet, and Sortformer build independently of
`MereRunCore`; the speech orchestration modules retain compatibility exports.

Speech synthesis command path:

- CLI: `Sources/MereRunCLI/Commands/SpeechSynthesizeCommand.swift`
- Runtime entry point: `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator.swift`
- Read next:
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Loading.swift`
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Generation.swift`
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+PromptPreparation.swift`
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+TokenGeneration.swift`
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+StreamingAudio.swift`
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Support.swift`

Speech tokenizer internals:

- Speech-token tensor surface:
  `Sources/AudioQwen3TTSModel/Qwen3TTSSpeechTokenizer.swift`
- Decoder stack:
  `Sources/AudioQwen3TTSModel/Qwen3TTSSpeechTokenizer+Decoder.swift`
- Encoder stack:
  `Sources/AudioQwen3TTSModel/Qwen3TTSSpeechTokenizer+Encoder.swift`

Speech transcription:

- [Shared operation](./internals/speech-transcription-operation.md): `Sources/AudioCore/SpeechTranscriptionOperation.swift`
- Model resolution: `Sources/AudioSTT/SpeechTranscriptionResolver.swift`
- CLI: `Sources/MereRunCLI/Commands/SpeechTranscribeCommand.swift`
- Runtime roots:
  - `Sources/AudioSTT/Qwen3ASR/Qwen3ASRGenerator.swift`
  - `Sources/AudioSTT/Parakeet/ParakeetGenerator.swift`
- Loading and execution: the adjacent `+Loading.swift`, `+Generation.swift`,
  and `+Decoding.swift` files
- Qwen model layers: `Sources/AudioQwen3ASRModel`
- Parakeet model layers and decoders: `Sources/AudioParakeetModel`
- Shared cache protocol and implementations: `Sources/MereRunKVCache`

Speaker diarization:

- CLI: `Sources/MereRunCLI/Commands/SpeechDiarizeCommand.swift`
- Runtime root: `Sources/AudioSortformer/SortformerDiarizer.swift`
- Model and feature stack: `Sources/AudioSortformer/SortformerModel.swift`

## OCR and vision

OCR:

- CLI: `Sources/MereRunCLI/Commands/VisionOCRCommand.swift`
- Runtime entry point: `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator.swift`
- Read next:
  - `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator+Loading.swift`
  - `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator+Inference.swift`
  - `Sources/MereRunCore/LightOnOCR/LightOnOCRSupport.swift`

Captioning and inspect flows:

- CLI:
  - `Sources/MereRunCLI/Commands/VisionCaptionCommand.swift`
  - `Sources/MereRunCLI/Commands/VisionInspectCommand.swift`
  - `Sources/MereRunCLI/Commands/VisionSegmentCommand.swift`
- Runtime roots:
  - `Sources/MereRunCore/VLM/QwenVLCaptioner.swift`
  - `Sources/MereRunCore/VLM/Qwen3VLAutoCaptioner.swift`
  - `Sources/MereRunCore/VLM/QwenVLEncoder.swift`
  - `Sources/MereRunTextEncoder/Vision/QwenVisionAttention.swift`

Segmentation and tracking runtime:

- CLI:
  - `Sources/MereRunCLI/Commands/VisionSegmentCommand.swift`
  - `Sources/MereRunCLI/Commands/VisionTrackCommand.swift`
  - `Sources/MereRunCLI/Commands/VisionTrackLiveCommand.swift`
- Native runtime:
  - `Sources/MereRunCore/SAM3/SAM31Config.swift`
  - `Sources/MereRunCore/SAM3/SAM31Resources.swift`
  - `Sources/MereRunCore/SAM3/SAM31Tokenizer.swift`
  - `Sources/MereRunCore/SAM3/SAM31Model.swift`
  - `Sources/MereRunCore/SAM3/SAM31InteractiveSAM.swift`
  - `Sources/MereRunCore/SAM3/SAM31Prompts.swift`
  - `Sources/MereRunCore/SAM3/SAM31ImageSegmenter.swift`
  - `Sources/MereRunCore/SAM3/SAM31VideoIO.swift`
  - `Sources/MereRunCore/SAM3/SAM31VideoTracker.swift`
  - `Sources/MereRunCore/SAM3/SAM31CameraCapture.swift`

Native object reconstruction:

- CLI:
  - `Sources/MereRunCLI/Commands/VisionImageTo3DCommand.swift`
  - `Sources/MereRunCLI/Commands/VisionImageTo3DTrellis2Command.swift`
  - `Sources/MereRunCLI/Commands/ImageReconstruct3DMultiviewCommand.swift`
- Shared mesh contract: `Sources/MereRunCore/Asset3D`
- Native runtimes:
  - `Sources/MereRunCore/TripoSR/TripoSRGenerator.swift`
  - `Sources/MereRunCore/Trellis2/Trellis2Generator.swift`
  - `Sources/MereRunCore/InstantMesh/InstantMeshGenerator.swift`

## Music, SFX, and video

Music generation:

- CLI: `Sources/MereRunCLI/Commands/MusicGenerateCommand.swift`
- Runtime entry point: `Sources/MereRunCore/ACEStep/ACEStepPipeline.swift`
- Read next:
  - `Sources/MereRunCore/ACEStep/ACEStepPipeline+Prompting.swift`
  - `Sources/MereRunCore/ACEStep/ACEStepPipeline+Generation.swift`

Music source separation:

- CLI: `Sources/MereRunCLI/Commands/MusicSeparateCommand.swift`
- Runtime entry point: `Sources/MereRunCore/RoFormer/RoFormerSeparator.swift`
- Read next:
  - `Sources/MereRunCore/RoFormer/RoFormerResources.swift`
  - `Sources/MereRunCore/RoFormer/BSRoFormer.swift`
  - `Sources/MereRunCore/RoFormer/RoFormerDSP.swift`

Audio bandwidth extension and super-resolution:

- CLI: `Sources/MereRunCLI/Commands/AudioEnhanceCommand.swift`
- Runtime entry point: `Sources/MereRunCore/APBWE/APBWEEnhancer.swift`
- General-audio entry point: `Sources/MereRunCore/UniverSR/UniverSREnhancer.swift`
- Read next:
  - `Sources/MereRunCore/APBWE/APBWEResources.swift`
  - `Sources/MereRunCore/APBWE/APBWEModel.swift`
  - `Sources/MereRunCore/UniverSR/UniverSRResources.swift`
  - `Sources/MereRunCore/UniverSR/UniverSRModel.swift`
  - `docs/architecture/audio-enhancement-ap-bwe-report.md`
  - `docs/architecture/audio-enhancement-universr-report.md`

Sound-effect generation:

- CLI: `Sources/MereRunCLI/Commands/SFXGenerateCommand.swift`
- Runtime root: `Sources/MereRunCore/Woosh/WooshGenerator.swift`
- Read next:
  - `Sources/MereRunCore/Woosh/WooshDiT.swift`
  - `Sources/MereRunCore/Woosh/WooshRobertaTextEncoder.swift`
  - `Sources/MereRunCore/Woosh/WooshVocosDecoder.swift`

Video generation:

- CLI:
  - `Sources/MereRunCLI/Commands/VideoCommand.swift` (`VideoGenerate` and `VideoExportLatents`)
- Runtime state: `Sources/MereRunCore/LTX/LTXUnifiedAVGenerator.swift`
- Loading and generation: `LTXUnifiedAVGenerator+*.swift` in the same directory
- Model computation: `Sources/MereRunLTXModel/`
- Text encoding: `Sources/MereRunCore/LTX/LTXGemmaTextEncoder.swift`
- Output: `Sources/MereRunCore/LTX/LTXVideoMP4Writer.swift`

Read the generation options, actor state, loading extensions, and execution
stages before the model layers. The legacy distilled actor has separate loading
and generation extensions. See [LTX runtime boundaries](./internals/ltx-runtime-boundaries.md)
for ownership and validation.

Cosmos3-Edge omnimodal generation and world simulation:

- CLI generation and reasoner:
  `Sources/MereRunCLI/Commands/VideoCosmos3Command.swift`
- Persistent world server:
  `Sources/MereRunCLI/Commands/WorldCommand.swift`
- Runtime entry points:
  - `Sources/MereRunCore/Cosmos3/Cosmos3EdgeGenerator.swift`
  - `Sources/MereRunCore/Cosmos3/Cosmos3Reasoner.swift`
  - `Sources/MereRunCore/Cosmos3/Cosmos3WorldSession.swift`
- Read next:
  - `Sources/MereRunCore/Cosmos3/Cosmos3Resources.swift`
  - `Sources/MereRunCore/Cosmos3/Cosmos3Action.swift`
  - `Sources/MereRunCore/Cosmos3/Cosmos3Sequence.swift`
  - `Sources/MereRunCore/Cosmos3/Cosmos3Transformer.swift`
  - `Sources/MereRunCore/Cosmos3/Cosmos3ModelLoader.swift`
  - `Sources/MereRunCore/Cosmos3/Cosmos3Scheduler.swift`
  - `Sources/MereRunCore/Cosmos3/Cosmos3ReasonerVision.swift`

## Validation and diagnostics

The advanced image validator is intentionally separate from generation:

- CLI root: `Sources/MereRunCLI/Commands/ImageValidateCommand.swift`
- Read next:
  - `Sources/MereRunCLI/Commands/ImageValidateCommand+VAE.swift`
  - `Sources/MereRunCLI/Commands/ImageValidateCommand+EncoderTransformer.swift`
  - `Sources/MereRunCLI/Commands/ImageValidateCommand+Pipeline.swift`

## Contributor reading order

1. Start with `Sources/MereRunCLI/MereRunCLI.swift`.
2. Pick one modality and read the matching command file.
3. Jump to that modality's runtime entry point.
4. Follow companion files in order: loading or preparation, then generation,
   then output.
5. If you need implementation detail, continue to the larger model-definition
   files.
