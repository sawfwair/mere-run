# Architecture Reading Map

This repo is easiest to understand by starting at the CLI surface and then
following one runtime family at a time. The goal of this map is not to explain
every model detail, but to give contributors a reliable reading order.

For the broader documentation set, start at [mere.run Documentation](/).

## Start here

- CLI entrypoint: `Sources/MereRunCLI/MereRunCLI.swift`
- Shared CLI helpers: `Sources/MereRunCLI/Support`
- Model paths and manifests:
  - `Sources/MereRunCore/MereRunModelPaths.swift`
  - `Sources/MereRunCore/MereRunModelManifest.swift`
  - `Sources/MereRunCore/ModelResolver.swift`

If you want to understand what a command does end to end, start at the command
file, then jump to the family entrypoint listed below.

## Image families

Klein image generation:

- CLI: `Sources/MereRunCLI/Commands/ImageGenerateCommand.swift`
- Runtime entrypoint: `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator.swift`
- Read next:
  - `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+ModelLoading.swift`
  - `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Generation.swift`
  - `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Chat.swift`

ZImage generation:

- Runtime entrypoint: `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator.swift`
- Read next:
  - `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+ModelLoading.swift`
  - `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+Inference.swift`
  - `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+LoRA.swift`

Qwen image editing:

- Runtime entrypoint: `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator.swift`
- Read next:
  - `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+ModelLoading.swift`
  - `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+Encoding.swift`

Krea 2 generation and LoRA training:

- Runtime entrypoint: `Sources/MereRunCore/Krea2/Krea2Generator.swift`
- Read next:
  - `Sources/MereRunCore/Krea2/Krea2RawResources.swift`
  - `Sources/MereRunCore/Krea2/Krea2Resources.swift`
  - `Sources/MereRunCore/Krea2/Krea2LoRAInjector.swift`
  - `Sources/MereRunCore/Krea2/Krea2LoRATrainer.swift`
  - `Sources/MereRunCore/Krea2/Krea2Configs.swift`
  - `Sources/MereRunCore/Krea2/Krea2Model.swift`
  - `Sources/MereRunCore/Krea2/Krea2ModelLoader.swift`
  - `Sources/MereRunCore/Krea2/Krea2SampleBuilder.swift`

Shared text encoder stack used by image models:

- Public entrypoint:
  `Sources/MereRunCore/ZImageTurbo/Model/TextEncoder/TextEncoder.swift`
- Architecture internals:
  - `Sources/MereRunCore/ZImageTurbo/Model/TextEncoder/TextEncoder+RoPE.swift`
  - `Sources/MereRunCore/ZImageTurbo/Model/TextEncoder/TextEncoder+Blocks.swift`

## Speech stack

Speech synthesis command path:

- CLI: `Sources/MereRunCLI/Commands/SpeechSynthesizeCommand.swift`
- Runtime entrypoint: `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator.swift`
- Read next:
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Loading.swift`
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Generation.swift`
  - `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Support.swift`

Speech tokenizer internals:

- Public tokenizer surface:
  `Sources/AudioTTS/Qwen3TTS/Qwen3TTSSpeechTokenizer.swift`
- Decoder stack:
  `Sources/AudioTTS/Qwen3TTS/Qwen3TTSSpeechTokenizer+Decoder.swift`
- Encoder stack:
  `Sources/AudioTTS/Qwen3TTS/Qwen3TTSSpeechTokenizer+Encoder.swift`

Speech transcription:

- CLI: `Sources/MereRunCLI/Commands/SpeechTranscribeCommand.swift`
- Runtime roots:
  - `Sources/AudioSTT/Qwen3ASR/Qwen3ASRGenerator.swift`
  - `Sources/AudioSTT/Parakeet/ParakeetGenerator.swift`

Speaker diarization:

- CLI: `Sources/MereRunCLI/Commands/SpeechDiarizeCommand.swift`
- Runtime root: `Sources/AudioSTT/Sortformer/SortformerDiarizer.swift`
- Model and feature stack: `Sources/AudioSTT/Sortformer/SortformerModel.swift`

## OCR and vision

OCR:

- CLI: `Sources/MereRunCLI/Commands/VisionOCRCommand.swift`
- Runtime entrypoint: `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator.swift`
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
  - `Sources/MereRunCore/ZImageTurbo/Model/TextEncoder/Vision/QwenVisionAttention.swift`

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
- Runtime entrypoint: `Sources/MereRunCore/ACEStep/ACEStepPipeline.swift`
- Read next:
  - `Sources/MereRunCore/ACEStep/ACEStepPipeline+Prompting.swift`
  - `Sources/MereRunCore/ACEStep/ACEStepPipeline+Generation.swift`

Music source separation:

- CLI: `Sources/MereRunCLI/Commands/MusicSeparateCommand.swift`
- Runtime entrypoint: `Sources/MereRunCore/RoFormer/RoFormerSeparator.swift`
- Read next:
  - `Sources/MereRunCore/RoFormer/RoFormerResources.swift`
  - `Sources/MereRunCore/RoFormer/BSRoFormer.swift`
  - `Sources/MereRunCore/RoFormer/RoFormerDSP.swift`

Audio bandwidth extension and super-resolution:

- CLI: `Sources/MereRunCLI/Commands/AudioEnhanceCommand.swift`
- Runtime entrypoint: `Sources/MereRunCore/APBWE/APBWEEnhancer.swift`
- General-audio entrypoint: `Sources/MereRunCore/UniverSR/UniverSREnhancer.swift`
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
- Runtime root: `Sources/MereRunCore/LTX/LTXDistilledLatentGenerator.swift`
- Read next:
  - `Sources/MereRunCore/LTX/LTXGemmaTextEncoder.swift`
  - `Sources/MereRunCore/LTX/LTXVideoMP4Writer.swift`

The LTX runtime combines public generation flow with lower-level model
definitions. Read it in this order:

- public generation types and `LTXDistilledLatentGenerator`
- diffusion helpers such as `denoiseLoop`, latent conditioning, and image
  loading
- decoder and encoder support modules
- unified AV generator types near the end

Cosmos3-Edge omnimodal generation and world simulation:

- CLI generation and reasoner:
  `Sources/MereRunCLI/Commands/VideoCosmos3Command.swift`
- Persistent world server:
  `Sources/MereRunCLI/Commands/WorldCommand.swift`
- Runtime entrypoints:
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

## Suggested reading order for new contributors

1. Start with `Sources/MereRunCLI/MereRunCLI.swift`.
2. Pick one modality and read the matching command file.
3. Jump to that modality’s runtime entrypoint.
4. Follow companion files in order: loading or preparation, then generation,
   then output.
5. Only then drop into the larger model-definition files if you need
   implementation detail.
