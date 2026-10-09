# Clef structured decisions

`ClefDecisionOperation` owns local tokenizer loading, checkpoint loading, media
preparation, schema token spans, and SystemOne-style answer decoding. The
`ClefJointHead` and its evidence/decoder layers live in `MereRunQwenModel`, next
to the Qwen3.5 backbone and vision tower they consume. Inference uses Swift/MLX;
Python is used only by the independent fixture exporter. Both the 27B
`text-decide-clef-4bit` and 9B `text-decide-clef-flash-4bit` checkpoints use this
operation; backbone and head dimensions come from their pinned configurations.
Flash configuration provenance lives in `Fixtures/ClefFlash`.

The request uses a questions object whose source order is retained by a typed
JSON parser. Choice option IDs sort lexically; score levels retain array order;
noul options are true then false. Structured state and option semantics use
sorted, compact JSON. Only state is truncated to the token budget: complete
schema spans and the final assistant marker remain intact.

The operation is serial and owns its loaded modules until `unload()`. Each
prediction starts with fresh backbone state and retains every final hidden
state. The lexical prior gathers rows from the output embedding matrix and
dequantizes only those rows. No autoregressive generation or vocabulary logits
are allocated. The joint head is loaded strictly in BF16 and remains unquantized.

Images are local paths. Videos are arrays of local frame paths representing a
24-fps source, uniformly sampled at 2 fps with paired temporal patches. Processor
normalization, geometry, timestamps, placeholders, and multimodal rotary
positions follow the pinned reference. Mixed image/video records are rejected.
Media stays intact; requests that exceed the context budget fail before weights
load. Preflight decodes and resizes media but does not load weights or run MLX.

Run `swift test --filter Clef` and the repository gate. Fixture provenance lives
under `Tests/MereRunCoreTests/Fixtures/Clef`; reproduce it with
`scripts/fixtures/export-clef-reference.py` and
`scripts/fixtures/export-clef-media-reference.py`. Independent fixtures cover
the tiny FP32 head, schema encoding, Pillow RGB bicubic pixels, exact FP32
normalization, odd-frame video geometry, and BF16 learned-position interpolation.
The optional real-tokenizer test compares every ID
with the pinned reference; set `MERERUN_CLEF_TOKENIZER` to a directory containing
the checkpoint's `tokenizer.json` and `tokenizer_config.json` to run it.

Full-checkpoint text, image, video, and resized-image probes complete with matching
choice answers and probability differences within 0.0031 of the configured
reference. These probes do not establish general accuracy or exact full-model
parity. See
`docs/benchmarks/clef-native-qualification-2026-10-01.md` and its receipt for the
bounded results and remaining numerical difference.

Flash full-checkpoint parity is covered by the opt-in
`ClefFlashCheckpointParityTests`. Generate independent inputs with
`scripts/fixtures/export-clef-flash-parity.py`, then set
`MERERUN_TEST_MLX_DEVICE=gpu`, `MERERUN_TEST_CLEF_FLASH_ROOT`, and
`MERERUN_TEST_CLEF_PARITY_DIR`. This test calls the runtime directly and writes
`native-runtime.json` in the parity directory; it does not exercise CLI admission.
See `docs/benchmarks/clef-flash-native-qualification-2026-10-02.md` for results.

## Omni thinker

`ClefDecisionOperation` dispatches dense Clef/Flash to `ClefDenseDecisionOperation`
and Qwen3-Omni to `ClefOmniDecisionOperation` using the typed checkpoint identity.
Omni's MoE text computation and vision/deepstack layers belong to
`MereRunQwenModel`; its audio encoder belongs to `AudioQwen3ASRModel`. Core owns
strict shard mapping, request encoding, local media preparation, and orchestration.
Core now depends directly on `AudioQwen3ASRModel` because Omni needs its shared
Qwen audio tower without depending on speech-transcription orchestration.
The dependency closure also adds that model target to AudioTTS and its tests
through their existing Core edge; their direct dependencies remain unchanged.
Only thinker components are instantiated. Output-embedding rows are untied.

Omni permits mixed images, audio, and videos with a 64,000-token context. File
videos use the first decoded frame at or after each 2-fps timestamp, with a
262,144-pixel raster cap before processor resizing; frame arrays are already
sampled at 2 fps. File videos carry sound only when every video has a track.
Whisper features pad clips together before the centered STFT; feature masks
retain partial hops for shorter clips. Audio/video tokens interleave by time.
Local file decoding and raster scaling remain backend-dependent and require
full-checkpoint media qualification.

`Fixtures/ClefOmni` contains tiny untrained independent FP32 thinker, vision,
audio, joint-head, and Whisper preprocessing fixtures exported by
`scripts/fixtures/export-clef-omni-reference.py` with Transformers 5.10.2.
These checks establish component comparisons, not qualification of the original
BF16 checkpoint or general model quality. Eight short Q4 checkpoint probes now
pass on a 36 GB Mac; see `docs/benchmarks/clef-omni-q4-qualification-2026-10-09.md`
for measured allocation, probability differences, and qualification limits.

The optional Omni `quantization` descriptor permits only affine Q4/group-64
routed experts. The loader stacks packed weights, scales, and biases by expert
index, while ordinary thinker parameters remain BF16. A disk-loading regression
compares the packed thinker against its reconstructed dense counterpart.
The streaming converter omits speech-output tensors and preserves source/output
hashes; local memory and decision-quality qualification are separate steps.
