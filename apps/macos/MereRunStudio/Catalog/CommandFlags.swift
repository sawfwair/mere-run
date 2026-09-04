// Generated from MereRunCapabilityCatalog by
// MereRunStudioTests/CommandFlagsGenerationTests.swift. Do not edit by hand:
// run ./scripts/update-studio-command-flags.sh after changing the shared contract.

/// Every flag the shared contract declares, as a constant per capability.
///
/// `CommandArguments` builds argv from these instead of string literals, so a flag the
/// CLI renames or drops is a compile error in the app rather than a command line that
/// only fails when it runs.
enum CommandFlags {}

// MARK: - text chat

extension CommandFlags {
    /// `mere.run text chat` — Chat
    enum TextChat: CommandFlagNamespace {
        static let command = ["text", "chat"]
        static let defaultValues = [
            "--max-tokens": "2048",
            "--response-format": "text",
            "--lora-scale": "1.0"
        ]

        static let prompt = "--prompt"
        static let image = "--image"
        static let system = "--system"
        static let maxTokens = "--max-tokens"
        static let contextSize = "--context-size"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let topK = "--top-k"
        static let minP = "--min-p"
        static let kvBits = "--kv-bits"
        static let kvQuantScheme = "--kv-quant-scheme"
        static let kvGroupSize = "--kv-group-size"
        static let quantizedKVStart = "--quantized-kv-start"
        static let modelRoot = "--model-root"
        static let model = "--model"
        static let responseFormat = "--response-format"
        static let lora = "--lora"
        static let loraScale = "--lora-scale"
        static let thinking = "--thinking"
        static let noThinking = "--no-thinking"
        static let reasoningEffort = "--reasoning-effort"
        static let stats = "--stats"
        static let stream = "--stream"
        static let tools = "--tools"
        static let toolLoop = "--tool-loop"
        static let sandboxDir = "--sandbox-dir"
        static let allowShellExec = "--allow-shell-exec"
        static let allowAbsoluteToolPaths = "--allow-absolute-tool-paths"
        static let autoApproveTools = "--auto-approve-tools"
        static let quiet = "--quiet"
        static let preflight = "--preflight"
        static let json = "--json"
        static let requireInstalled = "--require-installed"
    }
}

// MARK: - text code

extension CommandFlags {
    /// `mere.run text code` — Code
    enum TextCode: CommandFlagNamespace {
        static let command = ["text", "code"]
        static let defaultValues = [
            "--system": "You are a helpful coding assistant.",
            "--max-tokens": "2048",
            "--temperature": "1.0",
            "--top-p": "0.95",
            "--min-p": "0.0"
        ]

        static let prompt = "--prompt"
        static let system = "--system"
        static let maxTokens = "--max-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let minP = "--min-p"
        static let model = "--model"
        static let stats = "--stats"
        static let quiet = "--quiet"
        static let stream = "--stream"
    }
}

// MARK: - text embed

extension CommandFlags {
    /// `mere.run text embed` — Embeddings
    enum TextEmbed: CommandFlagNamespace {
        static let command = ["text", "embed"]

        static let model = "--model"
        static let maxTokens = "--max-tokens"
        static let output = "--output"
        static let pretty = "--pretty"
    }
}

// MARK: - text anonymize

extension CommandFlags {
    /// `mere.run text anonymize` — Anonymize
    enum TextAnonymize: CommandFlagNamespace {
        static let command = ["text", "anonymize"]

        static let model = "--model"
        static let maxTokens = "--max-tokens"
        static let replacement = "--replacement"
        static let json = "--json"
        static let pretty = "--pretty"
        static let output = "--output"
    }
}

// MARK: - text train-lora

extension CommandFlags {
    /// `mere.run text train-lora` — Train text LoRA
    enum TextTrainLoRA: CommandFlagNamespace {
        static let command = ["text", "train-lora"]

        static let data = "--data"
        static let output = "--output"
        static let model = "--model"
        static let modelPath = "--model-path"
        static let eval = "--eval"
        static let adapterName = "--adapter-name"
        static let trainingSteps = "--training-steps"
        static let batchSize = "--batch-size"
        static let learningRate = "--learning-rate"
        static let rank = "--rank"
        static let alpha = "--alpha"
        static let maxSequenceLength = "--max-sequence-length"
        static let reasoningEffort = "--reasoning-effort"
        static let seed = "--seed"
        static let targetModules = "--target-modules"
        static let dryRun = "--dry-run"
        static let visualize = "--visualize"
        static let visualizePort = "--visualize-port"
        static let json = "--json"
    }
}

// MARK: - image generate

extension CommandFlags {
    /// `mere.run image generate` — Generate and edit images
    enum ImageGenerate: CommandFlagNamespace {
        static let command = ["image", "generate"]
        static let defaultValues = [
            "--width": "1024",
            "--height": "1024",
            "--mask-feather": "8",
            "--max-sequence-length": "512",
            "--structured-prompt-model": "text-chat-gemma4-12b-4bit",
            "--structured-prompt-max-tokens": "2048",
            "--lora-scale": "1.0"
        ]

        static let prompt = "--prompt"
        static let negativePrompt = "--negative-prompt"
        static let cfg = "--cfg"
        static let sigmaShift = "--sigma-shift"
        static let output = "--output"
        static let width = "--width"
        static let height = "--height"
        static let steps = "--steps"
        static let seed = "--seed"
        static let model = "--model"
        static let input = "--input"
        static let mask = "--mask"
        static let outpaint = "--outpaint"
        static let maskFeather = "--mask-feather"
        static let refImage = "--ref-image"
        static let keepOriginalAspect = "--keep-original-aspect"
        static let strength = "--strength"
        static let maxSequenceLength = "--max-sequence-length"
        static let structuredPrompt = "--structured-prompt"
        static let structuredPromptModel = "--structured-prompt-model"
        static let structuredPromptModelRoot = "--structured-prompt-model-root"
        static let structuredPromptMaxTokens = "--structured-prompt-max-tokens"
        static let structuredPromptOutput = "--structured-prompt-output"
        static let lora = "--lora"
        static let loraScale = "--lora-scale"
        static let kreaConditioningMultiplier = "--krea-conditioning-multiplier"
        static let kreaConditioningLayerWeights = "--krea-conditioning-layer-weights"
        static let kreaBaseQuantizationBits = "--krea-base-quantization-bits"
        static let preflight = "--preflight"
        static let json = "--json"
        static let quiet = "--quiet"
        static let progressJSON = "--progress-json"
        static let receipt = "--receipt"
    }
}

// MARK: - image train-lora

extension CommandFlags {
    /// `mere.run image train-lora` — Train image LoRA
    enum ImageTrainLoRA: CommandFlagNamespace {
        static let command = ["image", "train-lora"]

        static let data = "--data"
        static let output = "--output"
        static let model = "--model"
        static let width = "--width"
        static let height = "--height"
        static let trainingSteps = "--training-steps"
        static let batchSize = "--batch-size"
        static let learningRate = "--learning-rate"
        static let rank = "--rank"
        static let alpha = "--alpha"
        static let maxTextLength = "--max-text-length"
        static let schedulerSteps = "--scheduler-steps"
        static let captionDropout = "--caption-dropout"
        static let seed = "--seed"
        static let lite = "--lite"
        static let baseQuantizationBits = "--base-quantization-bits"
        static let excludePreviewImages = "--exclude-preview-images"
        static let checkpointInterval = "--checkpoint-interval"
        static let resumeFrom = "--resume-from"
        static let maxResolution = "--max-resolution"
        static let progressive = "--progressive"
        static let lowRam = "--low-ram"
        static let noCompile = "--no-compile"
        static let gradientCheckpointing = "--gradient-checkpointing"
        static let recipe = "--recipe"
        static let benchmarkSteps = "--benchmark-steps"
        static let benchmarkWarmupSteps = "--benchmark-warmup-steps"
        static let sampleInterval = "--sample-interval"
        static let samplePrompt = "--sample-prompt"
        static let sampleModel = "--sample-model"
        static let sampleSteps = "--sample-steps"
        static let sampleCfg = "--sample-cfg"
        static let sampleLoRAScale = "--sample-lora-scale"
        static let sampleSeed = "--sample-seed"
        static let visualize = "--visualize"
        static let visualizePort = "--visualize-port"
        static let preflight = "--preflight"
        static let json = "--json"
        static let loraTargetRanks = "--lora-target-ranks"
        static let loraRankPreset = "--lora-rank-preset"
        static let loraTargetPreset = "--lora-target-preset"
        static let loraTargetMode = "--lora-target-mode"
        static let timestepSampling = "--timestep-sampling"
        static let timestepLossWeighting = "--timestep-loss-weighting"
        static let lossWeighting = "--loss-weighting"
        static let timestepLow = "--timestep-low"
        static let timestepHigh = "--timestep-high"
        static let lrWarmupSteps = "--lr-warmup-steps"
        static let noCosineScheduler = "--no-cosine-scheduler"
        static let lrMinFactor = "--lr-min-factor"
        static let adamWeightDecay = "--adam-weight-decay"
        static let syntheticSamples = "--synthetic-samples"
        static let quiet = "--quiet"
    }
}

// MARK: - image validate

extension CommandFlags {
    /// `mere.run image validate` — Validate image runtime
    enum ImageValidate: CommandFlagNamespace {
        static let command = ["image", "validate"]

        static let test = "--test"
        static let family = "--family"
        static let output = "--output"
        static let saveReference = "--save-reference"
        static let compare = "--compare"
        static let referenceDir = "--reference-dir"
    }
}

// MARK: - image dataset discover

extension CommandFlags {
    /// `mere.run image dataset discover` — Discover image datasets
    enum ImageDatasetDiscover: CommandFlagNamespace {
        static let command = ["image", "dataset", "discover"]

        static let root = "--root"
        static let maxDepth = "--max-depth"
        static let minUsablePairs = "--min-usable-pairs"
        static let trainingOutputRoot = "--training-output-root"
        static let trainingModel = "--training-model"
        static let trainingRecipe = "--training-recipe"
        static let excludePreviewImages = "--exclude-preview-images"
        static let json = "--json"
    }
}

// MARK: - image run-plan

extension CommandFlags {
    /// `mere.run image run-plan` — Run image plan
    enum ImageRunPlan: CommandFlagNamespace {
        static let command = ["image", "run-plan"]

        static let preflight = "--preflight"
        static let json = "--json"
        static let materialize = "--materialize"
    }
}

// MARK: - image visualize-run

extension CommandFlags {
    /// `mere.run image visualize-run` — Visualize image run
    enum ImageVisualizeRun: CommandFlagNamespace {
        static let command = ["image", "visualize-run"]

        static let port = "--port"
    }
}

// MARK: - image reconstruct-3d

extension CommandFlags {
    /// `mere.run image reconstruct-3d` — TripoSR reconstruction
    enum ImageReconstruct3D: CommandFlagNamespace {
        static let command = ["image", "reconstruct-3d"]

        static let output = "--output"
        static let model = "--model"
        static let resolution = "--resolution"
        static let densityThreshold = "--density-threshold"
        static let foregroundRatio = "--foreground-ratio"
        static let alreadyFramed = "--already-framed"
        static let noVertexColors = "--no-vertex-colors"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - image reconstruct-3d-trellis2

extension CommandFlags {
    /// `mere.run image reconstruct-3d-trellis2` — TRELLIS.2 reconstruction
    enum ImageReconstruct3DTrellis2: CommandFlagNamespace {
        static let command = ["image", "reconstruct-3d-trellis2"]

        static let output = "--output"
        static let model = "--model"
        static let seed = "--seed"
        static let textureSeed = "--texture-seed"
        static let maxTokens = "--max-tokens"
        static let alreadyFramed = "--already-framed"
        static let noRemesh = "--no-remesh"
        static let remeshBand = "--remesh-band"
        static let sealRadius = "--seal-radius"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - image reconstruct-3d-multiview

extension CommandFlags {
    /// `mere.run image reconstruct-3d-multiview` — InstantMesh multiview reconstruction
    enum ImageReconstruct3DMultiview: CommandFlagNamespace {
        static let command = ["image", "reconstruct-3d-multiview"]

        static let view = "--view"
        static let output = "--output"
        static let model = "--model"
        static let cameras = "--cameras"
        static let resolution = "--resolution"
        static let noVertexColors = "--no-vertex-colors"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - vision embed

extension CommandFlags {
    /// `mere.run vision embed` — Multimodal embeddings
    enum VisionEmbed: CommandFlagNamespace {
        static let command = ["vision", "embed"]

        static let text = "--text"
        static let image = "--image"
        static let inputJSON = "--input-json"
        static let instruction = "--instruction"
        static let model = "--model"
        static let dimensions = "--dimensions"
        static let maxTokens = "--max-tokens"
        static let minPixels = "--min-pixels"
        static let maxPixels = "--max-pixels"
        static let output = "--output"
        static let pretty = "--pretty"
    }
}

// MARK: - vision inspect

extension CommandFlags {
    /// `mere.run vision inspect` — Inspect image
    enum VisionInspect: CommandFlagNamespace {
        static let command = ["vision", "inspect"]
        static let defaultValues = [
            "--max-tokens": "2048",
            "--temperature": "0.7",
            "--top-p": "0.9"
        ]

        static let prompt = "--prompt"
        static let model = "--model"
        static let maxTokens = "--max-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
    }
}

// MARK: - vision caption

extension CommandFlags {
    /// `mere.run vision caption` — Caption images
    enum VisionCaption: CommandFlagNamespace {
        static let command = ["vision", "caption"]
        static let defaultValues = [
            "--max-tokens": "96",
            "--temperature": "0.2",
            "--top-p": "0.9"
        ]

        static let model = "--model"
        static let outputDir = "--output-dir"
        static let prompt = "--prompt"
        static let promptFile = "--prompt-file"
        static let focus = "--focus"
        static let triggerToken = "--trigger-token"
        static let maxTokens = "--max-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
    }
}

// MARK: - vision ocr

extension CommandFlags {
    /// `mere.run vision ocr` — OCR
    enum VisionOCR: CommandFlagNamespace {
        static let command = ["vision", "ocr"]
        static let defaultValues = [
            "--backend": "lighton",
            "--model": "vision-ocr-lighton",
            "--glmocr-cli": "glmocr",
            "--infinity-runtime": "native",
            "--infinity-parser-cli": "parser",
            "--infinity-model": "vision-ocr-infinity-pro-int8",
            "--infinity-backend": "vllm-server",
            "--infinity-api-url": "http://localhost:8000/v1/chat/completions",
            "--infinity-api-key": "EMPTY",
            "--infinity-task": "doc2json",
            "--infinity-output-format": "md",
            "--infinity-batch-size": "1",
            "--infinity-min-pixels": "2048",
            "--infinity-max-pixels": "16777216",
            "--max-tokens": "4096",
            "--temperature": "0.2"
        ]

        static let backend = "--backend"
        static let compare = "--compare"
        static let model = "--model"
        static let glmocrCli = "--glmocr-cli"
        static let glmConfig = "--glm-config"
        static let infinityRuntime = "--infinity-runtime"
        static let infinityParserCli = "--infinity-parser-cli"
        static let infinityModel = "--infinity-model"
        static let infinityBackend = "--infinity-backend"
        static let infinityAPIURL = "--infinity-api-url"
        static let infinityAPIKey = "--infinity-api-key"
        static let infinityTask = "--infinity-task"
        static let infinityPrompt = "--infinity-prompt"
        static let infinityOutputFormat = "--infinity-output-format"
        static let infinityBatchSize = "--infinity-batch-size"
        static let infinityModelCacheDir = "--infinity-model-cache-dir"
        static let infinityMinPixels = "--infinity-min-pixels"
        static let infinityMaxPixels = "--infinity-max-pixels"
        static let outputDir = "--output-dir"
        static let maxTokens = "--max-tokens"
        static let temperature = "--temperature"
        static let quiet = "--quiet"
    }
}

// MARK: - vision ground

extension CommandFlags {
    /// `mere.run vision ground` — Ground objects
    enum VisionGround: CommandFlagNamespace {
        static let command = ["vision", "ground"]

        static let query = "--query"
        static let model = "--model"
        static let output = "--output"
        static let jsonOutput = "--json-output"
        static let maskOutputDir = "--mask-output-dir"
        static let preflight = "--preflight"
        static let json = "--json"
        static let quiet = "--quiet"
        static let receipt = "--receipt"
    }
}

// MARK: - vision segment

extension CommandFlags {
    /// `mere.run vision segment` — Segment objects
    enum VisionSegment: CommandFlagNamespace {
        static let command = ["vision", "segment"]
        static let defaultValues = [
            "--threshold": "0.05",
            "--resolution": "1008"
        ]

        static let prompt = "--prompt"
        static let box = "--box"
        static let point = "--point"
        static let model = "--model"
        static let output = "--output"
        static let jsonOutput = "--json-output"
        static let maskOutputDir = "--mask-output-dir"
        static let threshold = "--threshold"
        static let resolution = "--resolution"
        static let showBoxes = "--show-boxes"
        static let multimask = "--multimask"
        static let preflight = "--preflight"
        static let json = "--json"
        static let quiet = "--quiet"
        static let receipt = "--receipt"
    }
}

// MARK: - vision track

extension CommandFlags {
    /// `mere.run vision track` — Track objects
    enum VisionTrack: CommandFlagNamespace {
        static let command = ["vision", "track"]
        static let defaultValues = [
            "--init-frame": "0",
            "--threshold": "0.05",
            "--resolution": "1008"
        ]

        static let prompt = "--prompt"
        static let box = "--box"
        static let point = "--point"
        static let model = "--model"
        static let output = "--output"
        static let jsonOutput = "--json-output"
        static let maskOutputDir = "--mask-output-dir"
        static let initFrame = "--init-frame"
        static let endFrame = "--end-frame"
        static let threshold = "--threshold"
        static let resolution = "--resolution"
        static let showBoxes = "--show-boxes"
        static let showLabels = "--show-labels"
        static let preflight = "--preflight"
        static let json = "--json"
        static let quiet = "--quiet"
        static let receipt = "--receipt"
    }
}

// MARK: - vision track-live

extension CommandFlags {
    /// `mere.run vision track-live` — Track camera
    enum VisionTrackLive: CommandFlagNamespace {
        static let command = ["vision", "track-live"]

        static let prompt = "--prompt"
        static let model = "--model"
        static let output = "--output"
        static let jsonOutput = "--json-output"
        static let camera = "--camera"
        static let durationSeconds = "--duration-seconds"
        static let initFrame = "--init-frame"
        static let seedSearchFrames = "--seed-search-frames"
        static let threshold = "--threshold"
        static let resolution = "--resolution"
        static let showBoxes = "--show-boxes"
        static let showLabels = "--show-labels"
    }
}

// MARK: - vision face detect

extension CommandFlags {
    /// `mere.run vision face detect` — Detect faces
    enum VisionFaceDetect: CommandFlagNamespace {
        static let command = ["vision", "face", "detect"]

        static let model = "--model"
        static let scoreThreshold = "--score-threshold"
        static let executionProvider = "--execution-provider"
        static let jsonOutput = "--json-output"
        static let json = "--json"
        static let maxFaces = "--max-faces"
        static let includeEmbeddings = "--include-embeddings"
    }
}

// MARK: - vision face embed

extension CommandFlags {
    /// `mere.run vision face embed` — Embed face
    enum VisionFaceEmbed: CommandFlagNamespace {
        static let command = ["vision", "face", "embed"]

        static let model = "--model"
        static let scoreThreshold = "--score-threshold"
        static let executionProvider = "--execution-provider"
        static let jsonOutput = "--json-output"
        static let json = "--json"
        static let faceIndex = "--face-index"
    }
}

// MARK: - vision face compare

extension CommandFlags {
    /// `mere.run vision face compare` — Compare faces
    enum VisionFaceCompare: CommandFlagNamespace {
        static let command = ["vision", "face", "compare"]

        static let model = "--model"
        static let scoreThreshold = "--score-threshold"
        static let executionProvider = "--execution-provider"
        static let jsonOutput = "--json-output"
        static let json = "--json"
        static let referenceFaceIndex = "--reference-face-index"
        static let candidateFaceIndex = "--candidate-face-index"
    }
}

// MARK: - vision face batch

extension CommandFlags {
    /// `mere.run vision face batch` — Batch faces
    enum VisionFaceBatch: CommandFlagNamespace {
        static let command = ["vision", "face", "batch"]

        static let inputList = "--input-list"
        static let model = "--model"
        static let scoreThreshold = "--score-threshold"
        static let executionProvider = "--execution-provider"
        static let maxFaces = "--max-faces"
        static let includeEmbeddings = "--include-embeddings"
        static let jsonlOutput = "--jsonl-output"
        static let failFast = "--fail-fast"
    }
}

// MARK: - vision pose

extension CommandFlags {
    /// `mere.run vision pose` — Pose landmarks
    enum VisionPose: CommandFlagNamespace {
        static let command = ["vision", "pose"]

        static let jsonOutput = "--json-output"
        static let noBody = "--no-body"
        static let noHands = "--no-hands"
        static let noFace = "--no-face"
        static let maxHands = "--max-hands"
        static let minimumConfidence = "--minimum-confidence"
        static let json = "--json"
    }
}

// MARK: - vision flow

extension CommandFlags {
    /// `mere.run vision flow` — Optical flow
    enum VisionFlow: CommandFlagNamespace {
        static let command = ["vision", "flow"]

        static let output = "--output"
        static let jsonOutput = "--json-output"
        static let accuracy = "--accuracy"
        static let json = "--json"
    }
}

// MARK: - vision depth-video

extension CommandFlags {
    /// `mere.run vision depth-video` — Video depth
    enum VisionDepthVideo: CommandFlagNamespace {
        static let command = ["vision", "depth-video"]

        static let output = "--output"
        static let model = "--model"
        static let inputSize = "--input-size"
        static let maxFrames = "--max-frames"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - vision geometry

extension CommandFlags {
    /// `mere.run vision geometry` — Metric geometry
    enum VisionGeometry: CommandFlagNamespace {
        static let command = ["vision", "geometry"]

        static let output = "--output"
        static let model = "--model"
        static let resolutionLevel = "--resolution-level"
        static let tokenCount = "--token-count"
        static let maxPoints = "--max-points"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - vision geometry-multiview

extension CommandFlags {
    /// `mere.run vision geometry-multiview` — Multi-view geometry
    enum VisionGeometryMultiview: CommandFlagNamespace {
        static let command = ["vision", "geometry-multiview"]

        static let output = "--output"
        static let model = "--model"
        static let cameras = "--cameras"
        static let processResolution = "--process-resolution"
        static let referenceView = "--reference-view"
        static let confidencePercentile = "--confidence-percentile"
        static let maxPoints = "--max-points"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - audio enhance

extension CommandFlags {
    /// `mere.run audio enhance` — Enhance audio
    enum AudioEnhance: CommandFlagNamespace {
        static let command = ["audio", "enhance"]

        static let model = "--model"
        static let modelPath = "--model-path"
        static let output = "--output"
        static let overlap = "--overlap"
        static let inputRate = "--input-rate"
        static let odeMethod = "--ode-method"
        static let odeSteps = "--ode-steps"
        static let guidanceScale = "--guidance-scale"
        static let seed = "--seed"
        static let chunkSeconds = "--chunk-seconds"
        static let dtype = "--dtype"
        static let quiet = "--quiet"
    }
}

// MARK: - audio generate

extension CommandFlags {
    /// `mere.run audio generate` — Generate LTX audio
    enum AudioGenerate: CommandFlagNamespace {
        static let command = ["audio", "generate"]

        static let output = "--output"
        static let model = "--model"
        static let modelRoot = "--model-root"
        static let negativePrompt = "--negative-prompt"
        static let enhancePrompt = "--enhance-prompt"
        static let promptEnhancerModel = "--prompt-enhancer-model"
        static let promptEnhancerModelRoot = "--prompt-enhancer-model-root"
        static let duration = "--duration"
        static let autoDuration = "--auto-duration"
        static let numFrames = "--num-frames"
        static let fps = "--fps"
        static let steps = "--steps"
        static let seed = "--seed"
        static let audioCfgGuidanceScale = "--audio-cfg-guidance-scale"
        static let audioStgGuidanceScale = "--audio-stg-guidance-scale"
        static let audioRescale = "--audio-rescale"
        static let audioStgBlock = "--audio-stg-block"
        static let audioSkipStep = "--audio-skip-step"
        static let sigmas = "--sigmas"
        static let lora = "--lora"
        static let quiet = "--quiet"
    }
}

// MARK: - music generate

extension CommandFlags {
    /// `mere.run music generate` — Generate and edit music
    enum MusicGenerate: CommandFlagNamespace {
        static let command = ["music", "generate"]
        static let defaultValues = [
            "--export-format": "pcm24",
            "--normalize": "peak",
            "--target-peak-db": "-1.0",
            "--fade-in-ms": "5.0",
            "--fade-out-ms": "20.0",
            "--adapter-kind": "auto",
            "--model": "music-acestep",
            "--decoder-subdirectory": "acestep-v15-turbo",
            "--vae-subdirectory": "vae",
            "--audio-cover-strength": "1.0",
            "--cover-noise-strength": "0.0",
            "--retake-variance": "0.0",
            "--vocal-language": "en",
            "--instruction": "Fill the audio semantic mask based on the given conditions:",
            "--task-type": "text2music",
            "--repaint-start": "0.0",
            "--repaint-end": "-1.0",
            "--chunk-mask-mode": "auto",
            "--repaint-mode": "balanced",
            "--repaint-strength": "0.5",
            "--flow-edit-n-min": "0.0",
            "--flow-edit-n-max": "1.0",
            "--flow-edit-n-average": "1",
            "--lm-temperature": "0.85",
            "--lm-top-k": "0",
            "--lm-top-p": "0.9",
            "--lm-repetition-penalty": "1.0",
            "--lm-cfg-scale": "2.0",
            "--lm-negative-prompt": "NO USER INPUT",
            "--vae-chunk-size": "512",
            "--vae-overlap": "64",
            "--temperature": "1.0",
            "--style-conditioning": "streaming",
            "--top-k": "100",
            "--cfg-musiccoca": "3.0",
            "--cfg-notes": "5.0",
            "--cfg-drums": "1.0",
            "--unmask-width": "0",
            "--seed-rotation": "0",
            "--prefill-duration": "1.64"
        ]

        static let lyrics = "--lyrics"
        static let lyricsFile = "--lyrics-file"
        static let instrumental = "--instrumental"
        static let lrcFile = "--lrc-file"
        static let lrcOutput = "--lrc-output"
        static let output = "--output"
        static let exportFormat = "--export-format"
        static let normalize = "--normalize"
        static let targetPeakDb = "--target-peak-db"
        static let fadeInMs = "--fade-in-ms"
        static let fadeOutMs = "--fade-out-ms"
        static let noDither = "--no-dither"
        static let recipeOutput = "--recipe-output"
        static let noRecipe = "--no-recipe"
        static let dawBundle = "--daw-bundle"
        static let stems = "--stems"
        static let adapter = "--adapter"
        static let adapterKind = "--adapter-kind"
        static let adapterScale = "--adapter-scale"
        static let model = "--model"
        static let checkpointsRoot = "--checkpoints-root"
        static let decoderSubdirectory = "--decoder-subdirectory"
        static let vaeSubdirectory = "--vae-subdirectory"
        static let lmSubdirectory = "--lm-subdirectory"
        static let lmModel = "--lm-model"
        static let textSubdirectory = "--text-subdirectory"
        static let useLM = "--use-lm"
        static let noLM = "--no-lm"
        static let analyzeSourceAudio = "--analyze-source-audio"
        static let duration = "--duration"
        static let quality = "--quality"
        static let steps = "--steps"
        static let shift = "--shift"
        static let inferMethod = "--infer-method"
        static let sampler = "--sampler"
        static let guidanceScale = "--guidance-scale"
        static let guidanceMode = "--guidance-mode"
        static let cfgIntervalStart = "--cfg-interval-start"
        static let cfgIntervalEnd = "--cfg-interval-end"
        static let velocityNormThreshold = "--velocity-norm-threshold"
        static let velocityEmaFactor = "--velocity-ema-factor"
        static let seed = "--seed"
        static let candidates = "--candidates"
        static let keepCandidates = "--keep-candidates"
        static let audioCoverStrength = "--audio-cover-strength"
        static let coverNoiseStrength = "--cover-noise-strength"
        static let retakeSeed = "--retake-seed"
        static let retakeVariance = "--retake-variance"
        static let vocalLanguage = "--vocal-language"
        static let instruction = "--instruction"
        static let taskType = "--task-type"
        static let sourceAudio = "--source-audio"
        static let referenceAudio = "--reference-audio"
        static let trackName = "--track-name"
        static let completeTrackClasses = "--complete-track-classes"
        static let nonCover = "--non-cover"
        static let repaintStart = "--repaint-start"
        static let repaintEnd = "--repaint-end"
        static let chunkMaskMode = "--chunk-mask-mode"
        static let repaintMode = "--repaint-mode"
        static let repaintStrength = "--repaint-strength"
        static let flowEdit = "--flow-edit"
        static let sourceCaption = "--source-caption"
        static let sourceLyrics = "--source-lyrics"
        static let flowEditNMin = "--flow-edit-n-min"
        static let flowEditNMax = "--flow-edit-n-max"
        static let flowEditNAverage = "--flow-edit-n-average"
        static let bpm = "--bpm"
        static let keyscale = "--keyscale"
        static let timesignature = "--timesignature"
        static let lmTemperature = "--lm-temperature"
        static let lmTopK = "--lm-top-k"
        static let lmTopP = "--lm-top-p"
        static let lmRepetitionPenalty = "--lm-repetition-penalty"
        static let lmCfgScale = "--lm-cfg-scale"
        static let lmNegativePrompt = "--lm-negative-prompt"
        static let metadataDuration = "--metadata-duration"
        static let metadataLanguage = "--metadata-language"
        static let noTiledVAE = "--no-tiled-vae"
        static let vaeChunkSize = "--vae-chunk-size"
        static let vaeOverlap = "--vae-overlap"
        static let quiet = "--quiet"
        static let temperature = "--temperature"
        static let styleConditioning = "--style-conditioning"
        static let topK = "--top-k"
        static let cfgMusiccoca = "--cfg-musiccoca"
        static let cfgNotes = "--cfg-notes"
        static let cfgDrums = "--cfg-drums"
        static let drumless = "--drumless"
        static let unmaskWidth = "--unmask-width"
        static let seedRotation = "--seed-rotation"
        static let prefillSilence = "--prefill-silence"
        static let prefillDuration = "--prefill-duration"
        static let progressJSON = "--progress-json"
        static let receipt = "--receipt"
    }
}

// MARK: - music analyze

extension CommandFlags {
    /// `mere.run music analyze` — Analyze music
    enum MusicAnalyze: CommandFlagNamespace {
        static let command = ["music", "analyze"]

        static let model = "--model"
        static let checkpointsRoot = "--checkpoints-root"
        static let decoderSubdirectory = "--decoder-subdirectory"
        static let vaeSubdirectory = "--vae-subdirectory"
        static let lmSubdirectory = "--lm-subdirectory"
        static let lmModel = "--lm-model"
        static let duration = "--duration"
        static let maxNewTokens = "--max-new-tokens"
        static let lmTemperature = "--lm-temperature"
        static let lmTopK = "--lm-top-k"
        static let lmTopP = "--lm-top-p"
        static let includeRawLM = "--include-raw-lm"
        static let includeAudioCodes = "--include-audio-codes"
        static let quiet = "--quiet"
    }
}

// MARK: - music transcribe

extension CommandFlags {
    /// `mere.run music transcribe` — Transcribe music
    enum MusicTranscribe: CommandFlagNamespace {
        static let command = ["music", "transcribe"]

        static let model = "--model"
        static let modelPath = "--model-path"
        static let variant = "--variant"
        static let output = "--output"
        static let format = "--format"
        static let instruments = "--instruments"
        static let listInstruments = "--list-instruments"
        static let sampling = "--sampling"
        static let temperature = "--temperature"
        static let maxTokensPerChunk = "--max-tokens-per-chunk"
        static let strictEos = "--strict-eos"
        static let beamSize = "--beam-size"
        static let chunkBatchSize = "--chunk-batch-size"
        static let dtype = "--dtype"
        static let noMusicalContext = "--no-musical-context"
        static let contextOutput = "--context-output"
        static let quiet = "--quiet"
    }
}

// MARK: - music separate

extension CommandFlags {
    /// `mere.run music separate` — Separate or restore music
    enum MusicSeparate: CommandFlagNamespace {
        static let command = ["music", "separate"]

        static let model = "--model"
        static let modelPath = "--model-path"
        static let outputDir = "--output-dir"
        static let overlap = "--overlap"
        static let dtype = "--dtype"
        static let quiet = "--quiet"
    }
}

// MARK: - music realtime

extension CommandFlags {
    /// `mere.run music realtime` — Realtime music
    enum MusicRealtime: CommandFlagNamespace {
        static let command = ["music", "realtime"]

        static let model = "--model"
        static let duration = "--duration"
        static let output = "--output"
        static let noPlay = "--no-play"
        static let styleConditioning = "--style-conditioning"
        static let temperature = "--temperature"
        static let topK = "--top-k"
        static let cfgMusiccoca = "--cfg-musiccoca"
        static let cfgNotes = "--cfg-notes"
        static let cfgDrums = "--cfg-drums"
        static let drumless = "--drumless"
        static let unmaskWidth = "--unmask-width"
        static let seedRotation = "--seed-rotation"
        static let prefillSilence = "--prefill-silence"
        static let prefillDuration = "--prefill-duration"
        static let interactive = "--interactive"
        static let listMidiInputs = "--list-midi-inputs"
        static let midiMonitor = "--midi-monitor"
        static let midiLogEvents = "--midi-log-events"
        static let midiLogRaw = "--midi-log-raw"
        static let midiInput = "--midi-input"
        static let midiChannel = "--midi-channel"
        static let midiNoteOffset = "--midi-note-offset"
        static let midiCc = "--midi-cc"
        static let quiet = "--quiet"
    }
}

// MARK: - music train-adapter

extension CommandFlags {
    /// `mere.run music train-adapter` — Train music adapter
    enum MusicTrainAdapter: CommandFlagNamespace {
        static let command = ["music", "train-adapter"]

        static let model = "--model"
        static let dataset = "--dataset"
        static let output = "--output"
        static let kind = "--kind"
        static let rank = "--rank"
        static let alpha = "--alpha"
        static let factor = "--factor"
        static let steps = "--steps"
        static let learningRate = "--learning-rate"
        static let weightDecay = "--weight-decay"
        static let seed = "--seed"
        static let maxDuration = "--max-duration"
        static let checkpointsRoot = "--checkpoints-root"
        static let decoderSubdirectory = "--decoder-subdirectory"
        static let vaeSubdirectory = "--vae-subdirectory"
        static let textSubdirectory = "--text-subdirectory"
        static let logEvery = "--log-every"
    }
}

// MARK: - music serve

extension CommandFlags {
    /// `mere.run music serve` — Serve resident music
    enum MusicServe: CommandFlagNamespace {
        static let command = ["music", "serve"]

        static let host = "--host"
        static let port = "--port"
        static let model = "--model"
        static let checkpointsRoot = "--checkpoints-root"
        static let decoderSubdirectory = "--decoder-subdirectory"
        static let vaeSubdirectory = "--vae-subdirectory"
        static let lmSubdirectory = "--lm-subdirectory"
        static let lmModel = "--lm-model"
        static let textSubdirectory = "--text-subdirectory"
        static let adapter = "--adapter"
        static let adapterKind = "--adapter-kind"
        static let adapterScale = "--adapter-scale"
        static let apiKey = "--api-key"
    }
}

// MARK: - video generate

extension CommandFlags {
    /// `mere.run video generate` — Generate video
    enum VideoGenerate: CommandFlagNamespace {
        static let command = ["video", "generate"]
        static let defaultValues = [
            "--spatial-overlap": "256",
            "--h3-weight-mode": "auto",
            "--h3-acceleration": "quality",
            "--guidance-scale": "5.0",
            "--shift": "5.0",
            "--audio-start-time": "0.0",
            "--a2v-guidance-scale": "3.0",
            "--video-cfg-guidance-scale": "3.0",
            "--audio-cfg-guidance-scale": "7.0",
            "--v2a-guidance-scale": "3.0",
            "--a2v-steps": "30",
            "--ltx-preset": "standard",
            "--ltx-pipeline": "two-stage",
            "--ltx-sampler-eta": "0.5",
            "--video-stg-scale": "1.0",
            "--video-guidance-rescale": "0.7",
            "--video-guidance-skip-step": "0",
            "--audio-stg-scale": "1.0",
            "--audio-guidance-rescale": "0.7",
            "--audio-guidance-skip-step": "0",
            "--res2s-bong-max-iterations": "100",
            "--gradient-estimation-gamma": "2.0",
            "--image-strength": "1.0",
            "--end-image-strength": "1.0",
            "--num-generated-keyframes": "0",
            "--conditioning-attention-strength": "1.0",
            "--temporal-upsample-rounds": "0"
        ]

        static let output = "--output"
        static let model = "--model"
        static let quality = "--quality"
        static let outputMode = "--output-mode"
        static let modelRoot = "--model-root"
        static let autoDuration = "--auto-duration"
        static let videoDecoder = "--video-decoder"
        static let hdr = "--hdr"
        static let hdrTransfer = "--hdr-transfer"
        static let highQualityHdr = "--high-quality-hdr"
        static let textEmbeddings = "--text-embeddings"
        static let spatialTile = "--spatial-tile"
        static let spatialOverlap = "--spatial-overlap"
        static let skipMp4 = "--skip-mp4"
        static let width = "--width"
        static let height = "--height"
        static let numFrames = "--num-frames"
        static let duration = "--duration"
        static let fps = "--fps"
        static let seed = "--seed"
        static let steps = "--steps"
        static let h3WeightMode = "--h3-weight-mode"
        static let h3Acceleration = "--h3-acceleration"
        static let guidanceScale = "--guidance-scale"
        static let shift = "--shift"
        static let negativePrompt = "--negative-prompt"
        static let enhancePrompt = "--enhance-prompt"
        static let promptEnhancerModel = "--prompt-enhancer-model"
        static let promptEnhancerModelRoot = "--prompt-enhancer-model-root"
        static let audio = "--audio"
        static let audioStartTime = "--audio-start-time"
        static let audioMaxDuration = "--audio-max-duration"
        static let a2vGuidanceScale = "--a2v-guidance-scale"
        static let videoCfgGuidanceScale = "--video-cfg-guidance-scale"
        static let audioCfgGuidanceScale = "--audio-cfg-guidance-scale"
        static let v2aGuidanceScale = "--v2a-guidance-scale"
        static let a2vSteps = "--a2v-steps"
        static let ltxPreset = "--ltx-preset"
        static let ltxPipeline = "--ltx-pipeline"
        static let ltxSampler = "--ltx-sampler"
        static let ltxSigmas = "--ltx-sigmas"
        static let ltxStage2Sigmas = "--ltx-stage-2-sigmas"
        static let distilledLoRAStrengthStage1 = "--distilled-lora-strength-stage-1"
        static let distilledLoRAStrengthStage2 = "--distilled-lora-strength-stage-2"
        static let ltxSamplerEta = "--ltx-sampler-eta"
        static let videoStgScale = "--video-stg-scale"
        static let videoGuidanceRescale = "--video-guidance-rescale"
        static let videoStgBlock = "--video-stg-block"
        static let videoGuidanceSkipStep = "--video-guidance-skip-step"
        static let audioStgScale = "--audio-stg-scale"
        static let audioGuidanceRescale = "--audio-guidance-rescale"
        static let audioStgBlock = "--audio-stg-block"
        static let audioGuidanceSkipStep = "--audio-guidance-skip-step"
        static let noRes2sBongMath = "--no-res2s-bong-math"
        static let res2sBongMaxIterations = "--res2s-bong-max-iterations"
        static let gradientEstimationGamma = "--gradient-estimation-gamma"
        static let image = "--image"
        static let imageStrength = "--image-strength"
        static let endImage = "--end-image"
        static let endImageStrength = "--end-image-strength"
        static let imageConditioning = "--image-conditioning"
        static let numGeneratedKeyframes = "--num-generated-keyframes"
        static let generatedKeyframe = "--generated-keyframe"
        static let lora = "--lora"
        static let videoConditioning = "--video-conditioning"
        static let conditioningAttentionStrength = "--conditioning-attention-strength"
        static let conditioningAttentionMask = "--conditioning-attention-mask"
        static let skipStage2 = "--skip-stage-2"
        static let referenceDownscaleFactor = "--reference-downscale-factor"
        static let referenceTemporalScaleFactor = "--reference-temporal-scale-factor"
        static let dfr = "--dfr"
        static let temporalUpsampleRounds = "--temporal-upsample-rounds"
        static let detailingLoRA = "--detailing-lora"
        static let detailingReferenceDownscaleFactor = "--detailing-reference-downscale-factor"
        static let reference = "--reference"
        static let preflight = "--preflight"
        static let json = "--json"
        static let timings = "--timings"
        static let timingsOutput = "--timings-output"
        static let quiet = "--quiet"
        static let progressJSON = "--progress-json"
        static let receipt = "--receipt"
    }
}

// MARK: - video retake

extension CommandFlags {
    /// `mere.run video retake` — Retake video region
    enum VideoRetake: CommandFlagNamespace {
        static let command = ["video", "retake"]

        static let source = "--source"
        static let frameRate = "--frame-rate"
        static let startTime = "--start-time"
        static let endTime = "--end-time"
        static let model = "--model"
        static let modelRoot = "--model-root"
        static let output = "--output"
        static let seed = "--seed"
        static let negativePrompt = "--negative-prompt"
        static let enhancePrompt = "--enhance-prompt"
        static let promptEnhancerModel = "--prompt-enhancer-model"
        static let promptEnhancerModelRoot = "--prompt-enhancer-model-root"
        static let steps = "--steps"
        static let sigmas = "--sigmas"
        static let lora = "--lora"
        static let videoCfgGuidanceScale = "--video-cfg-guidance-scale"
        static let videoStgScale = "--video-stg-scale"
        static let videoGuidanceRescale = "--video-guidance-rescale"
        static let videoModalityScale = "--video-modality-scale"
        static let videoStgBlock = "--video-stg-block"
        static let videoGuidanceSkipStep = "--video-guidance-skip-step"
        static let audioCfgGuidanceScale = "--audio-cfg-guidance-scale"
        static let audioStgScale = "--audio-stg-scale"
        static let audioGuidanceRescale = "--audio-guidance-rescale"
        static let audioModalityScale = "--audio-modality-scale"
        static let audioStgBlock = "--audio-stg-block"
        static let audioGuidanceSkipStep = "--audio-guidance-skip-step"
        static let videoDecoder = "--video-decoder"
        static let hdr = "--hdr"
        static let hdrTransfer = "--hdr-transfer"
        static let preserveVideo = "--preserve-video"
        static let preserveAudio = "--preserve-audio"
        static let quiet = "--quiet"
    }
}

// MARK: - video dub-it

extension CommandFlags {
    /// `mere.run video dub-it` — Dub-It
    enum VideoDubIt: CommandFlagNamespace {
        static let command = ["video", "dub-it"]

        static let referenceVideo = "--reference-video"
        static let icLoRA = "--ic-lora"
        static let icLoRAStrength = "--ic-lora-strength"
        static let referenceStrength = "--reference-strength"
        static let model = "--model"
        static let modelRoot = "--model-root"
        static let output = "--output"
        static let width = "--width"
        static let height = "--height"
        static let seed = "--seed"
        static let imageConditioning = "--image-conditioning"
        static let stage1Sigmas = "--stage-1-sigmas"
        static let stage2Sigmas = "--stage-2-sigmas"
        static let enhancePrompt = "--enhance-prompt"
        static let promptEnhancerModel = "--prompt-enhancer-model"
        static let promptEnhancerModelRoot = "--prompt-enhancer-model-root"
        static let videoDecoder = "--video-decoder"
        static let quiet = "--quiet"
    }
}

// MARK: - video animate

extension CommandFlags {
    /// `mere.run video animate` — Animate subject
    enum VideoAnimate: CommandFlagNamespace {
        static let command = ["video", "animate"]

        static let reference = "--reference"
        static let referenceMask = "--reference-mask"
        static let drivingVideo = "--driving-video"
        static let drivingMask = "--driving-mask"
        static let additionalReference = "--additional-reference"
        static let additionalReferenceMask = "--additional-reference-mask"
        static let output = "--output"
        static let model = "--model"
        static let modelRoot = "--model-root"
        static let mode = "--mode"
        static let profile = "--profile"
        static let width = "--width"
        static let height = "--height"
        static let steps = "--steps"
        static let guidanceScale = "--guidance-scale"
        static let shift = "--shift"
        static let sampler = "--sampler"
        static let distilledAdapter = "--distilled-adapter"
        static let distilledAdapterStrength = "--distilled-adapter-strength"
        static let seed = "--seed"
        static let fps = "--fps"
        static let segmentLength = "--segment-length"
        static let segmentOverlap = "--segment-overlap"
        static let tailPolicy = "--tail-policy"
        static let audioSource = "--audio-source"
        static let negativePrompt = "--negative-prompt"
        static let preflight = "--preflight"
        static let json = "--json"
        static let quiet = "--quiet"
    }
}

// MARK: - video cosmos3

extension CommandFlags {
    /// `mere.run video cosmos3` — Cosmos3
    enum VideoCosmos3: CommandFlagNamespace {
        static let command = ["video", "cosmos3"]

        static let mode = "--mode"
        static let model = "--model"
        static let output = "--output"
        static let actionsOutput = "--actions-output"
        static let image = "--image"
        static let video = "--video"
        static let negativePrompt = "--negative-prompt"
        static let width = "--width"
        static let height = "--height"
        static let numFrames = "--num-frames"
        static let steps = "--steps"
        static let guidanceScale = "--guidance-scale"
        static let shift = "--shift"
        static let schedule = "--schedule"
        static let seed = "--seed"
        static let fps = "--fps"
        static let conditionLatentFrame = "--condition-latent-frame"
        static let keepVideoTail = "--keep-video-tail"
        static let actionDomain = "--action-domain"
        static let actionFile = "--action-file"
        static let actionChunkSize = "--action-chunk-size"
        static let actionResolution = "--action-resolution"
        static let actionViewpoint = "--action-viewpoint"
        static let maxNewTokens = "--max-new-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let maxVideoFrames = "--max-video-frames"
        static let quiet = "--quiet"
    }
}

// MARK: - video prepare-masks

extension CommandFlags {
    /// `mere.run video prepare-masks` — Prepare SCAIL-2 masks
    enum VideoPrepareMasks: CommandFlagNamespace {
        static let command = ["video", "prepare-masks"]

        static let plan = "--plan"
        static let outputDir = "--output-dir"
        static let previewFrame = "--preview-frame"
        static let model = "--model"
        static let preflight = "--preflight"
        static let json = "--json"
        static let quiet = "--quiet"
    }
}

// MARK: - video export-latents

extension CommandFlags {
    /// `mere.run video export-latents` — Export video latents
    enum VideoExportLatents: CommandFlagNamespace {
        static let command = ["video", "export-latents"]

        static let model = "--model"
        static let modelRoot = "--model-root"
        static let output = "--output"
        static let width = "--width"
        static let height = "--height"
        static let numFrames = "--num-frames"
        static let seed = "--seed"
        static let quiet = "--quiet"
    }
}

// MARK: - video session

extension CommandFlags {
    /// `mere.run video session` — Resident LTX session
    enum VideoSession: CommandFlagNamespace {
        static let command = ["video", "session"]

        static let model = "--model"
        static let modelRoot = "--model-root"
        static let quiet = "--quiet"
    }
}

// MARK: - adapter list

extension CommandFlags {
    /// `mere.run adapter list` — Browse adapters
    enum AdapterList: CommandFlagNamespace {
        static let command = ["adapter", "list"]

        static let json = "--json"
    }
}

// MARK: - adapter pull

extension CommandFlags {
    /// `mere.run adapter pull` — Pull adapter
    enum AdapterPull: CommandFlagNamespace {
        static let command = ["adapter", "pull"]

        static let force = "--force"
        static let quiet = "--quiet"
    }
}

// MARK: - run list

extension CommandFlags {
    /// `mere.run run list` — Browse durable runs
    enum RunList: CommandFlagNamespace {
        static let command = ["run", "list"]

        static let root = "--root"
        static let executor = "--executor"
        static let limit = "--limit"
        static let maxDepth = "--max-depth"
        static let json = "--json"
    }
}

// MARK: - run inspect

extension CommandFlags {
    /// `mere.run run inspect` — Inspect durable run
    enum RunInspect: CommandFlagNamespace {
        static let command = ["run", "inspect"]

        static let json = "--json"
    }
}

// MARK: - run watch

extension CommandFlags {
    /// `mere.run run watch` — Watch remote run
    enum RunWatch: CommandFlagNamespace {
        static let command = ["run", "watch"]

        static let pollInterval = "--poll-interval"
        static let jsonStream = "--json-stream"
        static let json = "--json"
    }
}

// MARK: - run fetch

extension CommandFlags {
    /// `mere.run run fetch` — Fetch remote run
    enum RunFetch: CommandFlagNamespace {
        static let command = ["run", "fetch"]

        static let into = "--into"
        static let allArtifacts = "--all-artifacts"
        static let artifact = "--artifact"
        static let json = "--json"
    }
}

// MARK: - run cancel

extension CommandFlags {
    /// `mere.run run cancel` — Cancel run
    enum RunCancel: CommandFlagNamespace {
        static let command = ["run", "cancel"]

        static let json = "--json"
    }
}

// MARK: - run retry

extension CommandFlags {
    /// `mere.run run retry` — Retry Relay run
    enum RunRetry: CommandFlagNamespace {
        static let command = ["run", "retry"]

        static let json = "--json"
    }
}

// MARK: - eval pack validate

extension CommandFlags {
    /// `mere.run eval pack validate` — Validate evaluation pack
    enum EvalPackValidate: CommandFlagNamespace {
        static let command = ["eval", "pack", "validate"]

        static let json = "--json"
    }
}

// MARK: - eval run

extension CommandFlags {
    /// `mere.run eval run` — Run external evaluation
    enum EvalRun: CommandFlagNamespace {
        static let command = ["eval", "run"]

        static let model = "--model"
        static let adapter = "--adapter"
        static let trials = "--trials"
        static let maxTokens = "--max-tokens"
        static let contextSize = "--context-size"
        static let logprobs = "--logprobs"
        static let topLogprobs = "--top-logprobs"
        static let allowExternalScorer = "--allow-external-scorer"
        static let logResponses = "--log-responses"
        static let dryRun = "--dry-run"
        static let checkpoint = "--checkpoint"
        static let resume = "--resume"
        static let caseTrialLimit = "--case-trial-limit"
        static let output = "--output"
        static let json = "--json"
    }
}

// MARK: - eval promote

extension CommandFlags {
    /// `mere.run eval promote` — Promote evaluated artifact
    enum EvalPromote: CommandFlagNamespace {
        static let command = ["eval", "promote"]

        static let output = "--output"
        static let json = "--json"
    }
}

// MARK: - world serve

extension CommandFlags {
    /// `mere.run world serve` — World session
    enum WorldServe: CommandFlagNamespace {
        static let command = ["world", "serve"]

        static let host = "--host"
        static let port = "--port"
        static let apiKey = "--api-key"
        static let backend = "--backend"
        static let baseModel = "--base-model"
        static let model = "--model"
        static let stateDirectory = "--state-directory"
        static let prepare = "--prepare"
    }
}

// MARK: - vision serve

extension CommandFlags {
    /// `mere.run vision serve` — Vision grounding server
    enum VisionServe: CommandFlagNamespace {
        static let command = ["vision", "serve"]

        static let host = "--host"
        static let port = "--port"
        static let model = "--model"
        static let apiKey = "--api-key"
        static let maxFrameBytes = "--max-frame-bytes"
        static let maxBatchSize = "--max-batch-size"
        static let maxBatchBytes = "--max-batch-bytes"
        static let preflight = "--preflight"
        static let json = "--json"
    }
}

// MARK: - status

extension CommandFlags {
    /// `mere.run status` — Status snapshot
    enum Status: CommandFlagNamespace {
        static let command = ["status"]

        static let host = "--host"
        static let port = "--port"
        static let apiKey = "--api-key"
        static let timeoutSeconds = "--timeout-seconds"
        static let json = "--json"
    }
}

// MARK: - gate

extension CommandFlags {
    /// `mere.run gate` — Quality gate
    enum Gate: CommandFlagNamespace {
        static let command = ["gate"]

        static let suite = "--suite"
        static let updateBaselines = "--update-baselines"
        static let strictPerf = "--strict-perf"
        static let jsonOutput = "--json-output"
        static let list = "--list"
    }
}

// MARK: - model storage

extension CommandFlags {
    /// `mere.run model storage` — Model storage
    enum ModelStorage: CommandFlagNamespace {
        static let command = ["model", "storage"]

        static let json = "--json"
    }
}

// MARK: - model location list

extension CommandFlags {
    /// `mere.run model location list` — List model locations
    enum ModelLocationList: CommandFlagNamespace {
        static let command = ["model", "location", "list"]

        static let json = "--json"
    }
}

// MARK: - model location add

extension CommandFlags {
    /// `mere.run model location add` — Add search root
    enum ModelLocationAdd: CommandFlagNamespace {
        static let command = ["model", "location", "add"]
    }
}

// MARK: - model location remove

extension CommandFlags {
    /// `mere.run model location remove` — Remove search root
    enum ModelLocationRemove: CommandFlagNamespace {
        static let command = ["model", "location", "remove"]
    }
}

// MARK: - model location bind

extension CommandFlags {
    /// `mere.run model location bind` — Bind model directory
    enum ModelLocationBind: CommandFlagNamespace {
        static let command = ["model", "location", "bind"]

        static let acceptModelLicense = "--accept-model-license"
    }
}

// MARK: - model location unbind

extension CommandFlags {
    /// `mere.run model location unbind` — Unbind model directory
    enum ModelLocationUnbind: CommandFlagNamespace {
        static let command = ["model", "location", "unbind"]
    }
}

// MARK: - model gc

extension CommandFlags {
    /// `mere.run model gc` — Model storage cleanup
    enum ModelGc: CommandFlagNamespace {
        static let command = ["model", "gc"]

        static let force = "--force"
        static let json = "--json"
    }
}

// MARK: - model runtime get

extension CommandFlags {
    /// `mere.run model runtime get` — Read runtime policy
    enum ModelRuntimeGet: CommandFlagNamespace {
        static let command = ["model", "runtime", "get"]

        static let json = "--json"
    }
}

// MARK: - model runtime set

extension CommandFlags {
    /// `mere.run model runtime set` — Set runtime policy
    enum ModelRuntimeSet: CommandFlagNamespace {
        static let command = ["model", "runtime", "set"]

        static let alias = "--alias"
        static let clearAlias = "--clear-alias"
        static let pinned = "--pinned"
        static let unpinned = "--unpinned"
        static let ttlSeconds = "--ttl-seconds"
        static let clearTTL = "--clear-ttl"
        static let maxContextTokens = "--max-context-tokens"
        static let clearMaxContextTokens = "--clear-max-context-tokens"
        static let maxTokens = "--max-tokens"
        static let clearMaxTokens = "--clear-max-tokens"
        static let temperature = "--temperature"
        static let clearTemperature = "--clear-temperature"
        static let topP = "--top-p"
        static let clearTopP = "--clear-top-p"
        static let minP = "--min-p"
        static let clearMinP = "--clear-min-p"
        static let engine = "--engine"
        static let clearEngine = "--clear-engine"
        static let kvCacheMode = "--kv-cache-mode"
        static let clearKVCacheMode = "--clear-kv-cache-mode"
        static let json = "--json"
    }
}

// MARK: - setup

extension CommandFlags {
    /// `mere.run setup` — Setup path
    enum Setup: CommandFlagNamespace {
        static let command = ["setup"]

        static let mode = "--mode"
        static let agentModel = "--agent-model"
        static let install = "--install"
        static let start = "--start"
        static let dryRun = "--dry-run"
        static let host = "--host"
        static let port = "--port"
        static let piPath = "--pi-path"
        static let quiet = "--quiet"
    }
}

// MARK: - agent onboard

extension CommandFlags {
    /// `mere.run agent onboard` — Agent onboarding
    enum AgentOnboard: CommandFlagNamespace {
        static let command = ["agent", "onboard"]

        static let pullRecommended = "--pull-recommended"
        static let acceptModelLicense = "--accept-model-license"
        static let installPi = "--install-pi"
        static let configurePi = "--configure-pi"
        static let host = "--host"
        static let port = "--port"
        static let model = "--model"
        static let quiet = "--quiet"
    }
}

// MARK: - agent status

extension CommandFlags {
    /// `mere.run agent status` — Agent status
    enum AgentStatus: CommandFlagNamespace {
        static let command = ["agent", "status"]

        static let piPath = "--pi-path"
        static let json = "--json"
    }
}

// MARK: - agent install-pi

extension CommandFlags {
    /// `mere.run agent install-pi` — Install Pi
    enum AgentInstallPi: CommandFlagNamespace {
        static let command = ["agent", "install-pi"]

        static let force = "--force"
    }
}

// MARK: - agent start

extension CommandFlags {
    /// `mere.run agent start` — Start setup agent
    enum AgentStart: CommandFlagNamespace {
        static let command = ["agent", "start"]

        static let host = "--host"
        static let port = "--port"
        static let piPath = "--pi-path"
        static let prompt = "--prompt"
        static let model = "--model"
        static let skipServer = "--skip-server"
        static let allowUnsupported = "--allow-unsupported"
        static let noBootstrap = "--no-bootstrap"
        static let quiet = "--quiet"
    }
}

// MARK: - model list

extension CommandFlags {
    /// `mere.run model list` — List models
    enum ModelList: CommandFlagNamespace {
        static let command = ["model", "list"]
    }
}

// MARK: - model capabilities

extension CommandFlags {
    /// `mere.run model capabilities` — Model capabilities
    enum ModelCapabilities: CommandFlagNamespace {
        static let command = ["model", "capabilities"]

        static let all = "--all"
        static let recommended = "--recommended"
        static let json = "--json"
    }
}

// MARK: - model pull

extension CommandFlags {
    /// `mere.run model pull` — Pull model
    enum ModelPull: CommandFlagNamespace {
        static let command = ["model", "pull"]

        static let all = "--all"
        static let force = "--force"
        static let quiet = "--quiet"
        static let allowUnsupported = "--allow-unsupported"
        static let acceptModelLicense = "--accept-model-license"
        static let preflight = "--preflight"
        static let json = "--json"
    }
}

// MARK: - model info

extension CommandFlags {
    /// `mere.run model info` — Model info
    enum ModelInfo: CommandFlagNamespace {
        static let command = ["model", "info"]

        static let json = "--json"
        static let components = "--components"
    }
}

// MARK: - model remove

extension CommandFlags {
    /// `mere.run model remove` — Remove model
    enum ModelRemove: CommandFlagNamespace {
        static let command = ["model", "remove"]

        static let force = "--force"
        static let keepCache = "--keep-cache"
        static let json = "--json"
    }
}

// MARK: - model repair-manifests

extension CommandFlags {
    /// `mere.run model repair-manifests` — Repair manifests
    enum ModelRepairManifests: CommandFlagNamespace {
        static let command = ["model", "repair-manifests"]

        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - model optimize

extension CommandFlags {
    /// `mere.run model optimize` — Optimize model
    enum ModelOptimize: CommandFlagNamespace {
        static let command = ["model", "optimize"]

        static let force = "--force"
        static let output = "--output"
        static let json = "--json"
    }
}

// MARK: - model benchmark q36-mtp

extension CommandFlags {
    /// `mere.run model benchmark q36-mtp` — Qwen-family MTP benchmark
    enum ModelBenchmarkQ36MTP: CommandFlagNamespace {
        static let command = ["model", "benchmark", "q36-mtp"]

        static let model = "--model"
        static let modelRoot = "--model-root"
        static let prompt = "--prompt"
        static let promptFile = "--prompt-file"
        static let promptRepeat = "--prompt-repeat"
        static let promptRepeatValues = "--prompt-repeat-values"
        static let decodeTokens = "--decode-tokens"
        static let decodeTokenValues = "--decode-token-values"
        static let temperature = "--temperature"
        static let temperatureValues = "--temperature-values"
        static let topP = "--top-p"
        static let contextSize = "--context-size"
        static let mtpBlockSize = "--mtp-block-size"
        static let forcedMTPMinPromptTokens = "--forced-mtp-min-prompt-tokens"
        static let json = "--json"
    }
}

// MARK: - model benchmark laguna-dflash

extension CommandFlags {
    /// `mere.run model benchmark laguna-dflash` — Laguna DFlash benchmark
    enum ModelBenchmarkLagunaDFlash: CommandFlagNamespace {
        static let command = ["model", "benchmark", "laguna-dflash"]

        static let lagunaPath = "--laguna-path"
        static let lagunaDFlashPath = "--laguna-dflash-path"
        static let decodeTokenValues = "--decode-token-values"
        static let repetitions = "--repetitions"
        static let lagunaDFlashTokens = "--laguna-dflash-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let topK = "--top-k"
        static let minP = "--min-p"
        static let prompt = "--prompt"
        static let promptFile = "--prompt-file"
        static let fixture = "--fixture"
        static let contextSize = "--context-size"
        static let concurrencyValues = "--concurrency-values"
        static let warmupRepetitions = "--warmup-repetitions"
        static let mixedFixtures = "--mixed-fixtures"
        static let includeAutomatic = "--include-automatic"
        static let logResponses = "--log-responses"
        static let json = "--json"
    }
}

// MARK: - model benchmark chat

extension CommandFlags {
    /// `mere.run model benchmark chat` — Chat benchmark
    enum ModelBenchmarkChat: CommandFlagNamespace {
        static let command = ["model", "benchmark", "chat"]

        static let models = "--models"
        static let suite = "--suite"
        static let cases = "--cases"
        static let maxTokens = "--max-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let topK = "--top-k"
        static let minP = "--min-p"
        static let contextSize = "--context-size"
        static let concurrency = "--concurrency"
        static let lagunaPath = "--laguna-path"
        static let lagunaDFlashPath = "--laguna-dflash-path"
        static let lagunaDFlashTokens = "--laguna-dflash-tokens"
        static let lagunaDFlashMinTokens = "--laguna-dflash-min-tokens"
        static let lagunaDFlashRouting = "--laguna-dflash-routing"
        static let dryRun = "--dry-run"
        static let logResponses = "--log-responses"
        static let json = "--json"
    }
}

// MARK: - model benchmark code

extension CommandFlags {
    /// `mere.run model benchmark code` — Code benchmark
    enum ModelBenchmarkCode: CommandFlagNamespace {
        static let command = ["model", "benchmark", "code"]

        static let models = "--models"
        static let lagunaPath = "--laguna-path"
        static let lagunaDFlashPath = "--laguna-dflash-path"
        static let lagunaDFlashTokens = "--laguna-dflash-tokens"
        static let lagunaDFlashMinTokens = "--laguna-dflash-min-tokens"
        static let suite = "--suite"
        static let tasks = "--tasks"
        static let humanevalFile = "--humaneval-file"
        static let maxTokens = "--max-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let topK = "--top-k"
        static let minP = "--min-p"
        static let thinking = "--thinking"
        static let executionTimeout = "--execution-timeout"
        static let python = "--python"
        static let sandbox = "--sandbox"
        static let allowCodeExecution = "--allow-code-execution"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - model benchmark fused

extension CommandFlags {
    /// `mere.run model benchmark fused` — Fused quality suite
    enum ModelBenchmarkFused: CommandFlagNamespace {
        static let command = ["model", "benchmark", "fused"]

        static let suite = "--suite"
        static let models = "--models"
        static let manifest = "--manifest"
        static let externalCases = "--external-cases"
        static let cases = "--cases"
        static let capabilities = "--capabilities"
        static let trials = "--trials"
        static let maxTokens = "--max-tokens"
        static let contextSize = "--context-size"
        static let logprobs = "--logprobs"
        static let topLogprobs = "--top-logprobs"
        static let performanceLane = "--performance-lane"
        static let executionTimeout = "--execution-timeout"
        static let python = "--python"
        static let sandbox = "--sandbox"
        static let allowCodeExecution = "--allow-code-execution"
        static let logResponses = "--log-responses"
        static let checkpoint = "--checkpoint"
        static let resume = "--resume"
        static let caseTrialLimit = "--case-trial-limit"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - model benchmark fused-fixture

extension CommandFlags {
    /// `mere.run model benchmark fused-fixture` — Fused fixture hashes
    enum ModelBenchmarkFusedFixture: CommandFlagNamespace {
        static let command = ["model", "benchmark", "fused-fixture"]

        static let check = "--check"
    }
}

// MARK: - model benchmark vlm

extension CommandFlags {
    /// `mere.run model benchmark vlm` — Vision-language benchmark
    enum ModelBenchmarkVLM: CommandFlagNamespace {
        static let command = ["model", "benchmark", "vlm"]

        static let models = "--models"
        static let dataset = "--dataset"
        static let lmmsTasks = "--lmms-tasks"
        static let fixtureDir = "--fixture-dir"
        static let outputDir = "--output-dir"
        static let lmmsEvalRoot = "--lmms-eval-root"
        static let lmmsEvalPython = "--lmms-eval-python"
        static let externalEndpoint = "--external-endpoint"
        static let baseURL = "--base-url"
        static let apiKey = "--api-key"
        static let host = "--host"
        static let port = "--port"
        static let limit = "--limit"
        static let logSamples = "--log-samples"
        static let maxTokens = "--max-tokens"
        static let contextSize = "--context-size"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - model benchmark tool-calls

extension CommandFlags {
    /// `mere.run model benchmark tool-calls` — Tool-call benchmark
    enum ModelBenchmarkToolCalls: CommandFlagNamespace {
        static let command = ["model", "benchmark", "tool-calls"]

        static let models = "--models"
        static let cases = "--cases"
        static let maxTokens = "--max-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let topK = "--top-k"
        static let minP = "--min-p"
        static let contextSize = "--context-size"
        static let lagunaPath = "--laguna-path"
        static let lagunaDFlashPath = "--laguna-dflash-path"
        static let lagunaDFlashTokens = "--laguna-dflash-tokens"
        static let lagunaDFlashMinTokens = "--laguna-dflash-min-tokens"
        static let dryRun = "--dry-run"
        static let logResponses = "--log-responses"
        static let json = "--json"
    }
}

// MARK: - model benchmark tool-continuations

extension CommandFlags {
    /// `mere.run model benchmark tool-continuations` — Tool continuation benchmark
    enum ModelBenchmarkToolContinuations: CommandFlagNamespace {
        static let command = ["model", "benchmark", "tool-continuations"]

        static let model = "--model"
        static let modelRoot = "--model-root"
        static let maxTokens = "--max-tokens"
        static let contextSize = "--context-size"
        static let dryRun = "--dry-run"
        static let logResponses = "--log-responses"
        static let json = "--json"
    }
}

// MARK: - model benchmark gemma4-kv

extension CommandFlags {
    /// `mere.run model benchmark gemma4-kv` — Gemma4 KV benchmark
    enum ModelBenchmarkGemma4KV: CommandFlagNamespace {
        static let command = ["model", "benchmark", "gemma4-kv"]

        static let model = "--model"
        static let modelRoot = "--model-root"
        static let prompt = "--prompt"
        static let promptFile = "--prompt-file"
        static let promptRepeat = "--prompt-repeat"
        static let promptRepeatValues = "--prompt-repeat-values"
        static let decodeTokens = "--decode-tokens"
        static let decodeTokenValues = "--decode-token-values"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let json = "--json"
    }
}

// MARK: - model benchmark gemma4-mtp

extension CommandFlags {
    /// `mere.run model benchmark gemma4-mtp` — Gemma4 MTP benchmark
    enum ModelBenchmarkGemma4MTP: CommandFlagNamespace {
        static let command = ["model", "benchmark", "gemma4-mtp"]

        static let model = "--model"
        static let modelRoot = "--model-root"
        static let prompt = "--prompt"
        static let promptFile = "--prompt-file"
        static let promptRepeat = "--prompt-repeat"
        static let promptRepeatValues = "--prompt-repeat-values"
        static let decodeTokens = "--decode-tokens"
        static let decodeTokenValues = "--decode-token-values"
        static let mtpBlockSize = "--mtp-block-size"
        static let mtpMinPromptTokens = "--mtp-min-prompt-tokens"
        static let json = "--json"
    }
}

// MARK: - model benchmark api-workload

extension CommandFlags {
    /// `mere.run model benchmark api-workload` — API workload benchmark
    enum ModelBenchmarkAPIWorkload: CommandFlagNamespace {
        static let command = ["model", "benchmark", "api-workload"]

        static let host = "--host"
        static let port = "--port"
        static let apiKey = "--api-key"
        static let model = "--model"
        static let workloadFile = "--workload-file"
        static let turns = "--turns"
        static let sharedPrefixRepeat = "--shared-prefix-repeat"
        static let maxTokens = "--max-tokens"
        static let temperature = "--temperature"
        static let topP = "--top-p"
        static let concurrency = "--concurrency"
        static let timeoutSeconds = "--timeout-seconds"
        static let dryRun = "--dry-run"
        static let json = "--json"
    }
}

// MARK: - speech synthesize

extension CommandFlags {
    /// `mere.run speech synthesize` — Synthesize speech
    enum SpeechSynthesize: CommandFlagNamespace {
        static let command = ["speech", "synthesize"]
        static let defaultValues = [
            "--model": "speech-tts-qwen3-nano",
            "--voice": "A calm female voice with clear pronunciation",
            "--mode": "style",
            "--language": "auto",
            "--temperature": "0.6",
            "--stream-chunk-tokens": "25"
        ]

        static let output = "--output"
        static let model = "--model"
        static let voice = "--voice"
        static let mode = "--mode"
        static let profile = "--profile"
        static let refAudio = "--ref-audio"
        static let refText = "--ref-text"
        static let language = "--language"
        static let saveProfile = "--save-profile"
        static let temperature = "--temperature"
        static let stream = "--stream"
        static let streamChunkTokens = "--stream-chunk-tokens"
        static let quiet = "--quiet"
        static let progressJSON = "--progress-json"
        static let receipt = "--receipt"
    }
}

// MARK: - speech transcribe

extension CommandFlags {
    /// `mere.run speech transcribe` — Transcribe speech
    enum SpeechTranscribe: CommandFlagNamespace {
        static let command = ["speech", "transcribe"]
        static let defaultValues = [
            "--backend": "auto",
            "--task": "transcribe",
            "--max-tokens": "448",
            "--stream-chunk-ms": "200",
            "--stream-decode-ms": "2000"
        ]

        static let output = "--output"
        static let model = "--model"
        static let backend = "--backend"
        static let task = "--task"
        static let language = "--language"
        static let maxTokens = "--max-tokens"
        static let stream = "--stream"
        static let streamChunkMs = "--stream-chunk-ms"
        static let streamDecodeMs = "--stream-decode-ms"
        static let inputFormat = "--input-format"
        static let sampleRate = "--sample-rate"
        static let jsonl = "--jsonl"
        static let noTimestamps = "--no-timestamps"
        static let quiet = "--quiet"
        static let receipt = "--receipt"
    }
}

// MARK: - speech diarize

extension CommandFlags {
    /// `mere.run speech diarize` — Diarize speech
    enum SpeechDiarize: CommandFlagNamespace {
        static let command = ["speech", "diarize"]

        static let model = "--model"
        static let format = "--format"
        static let output = "--output"
        static let threshold = "--threshold"
        static let minDuration = "--min-duration"
        static let mergeGap = "--merge-gap"
        static let quiet = "--quiet"
    }
}

// MARK: - speech profile list

extension CommandFlags {
    /// `mere.run speech profile list` — Voice profiles
    enum SpeechProfileList: CommandFlagNamespace {
        static let command = ["speech", "profile", "list"]
    }
}

// MARK: - speech profile create

extension CommandFlags {
    /// `mere.run speech profile create` — Create voice profile
    enum SpeechProfileCreate: CommandFlagNamespace {
        static let command = ["speech", "profile", "create"]

        static let name = "--name"
        static let audio = "--audio"
        static let text = "--text"
        static let language = "--language"
        static let quiet = "--quiet"
    }
}

// MARK: - speech profile delete

extension CommandFlags {
    /// `mere.run speech profile delete` — Delete voice profile
    enum SpeechProfileDelete: CommandFlagNamespace {
        static let command = ["speech", "profile", "delete"]

        static let id = "--id"
    }
}

// MARK: - speech listen

extension CommandFlags {
    /// `mere.run speech listen` — Live transcription
    enum SpeechListen: CommandFlagNamespace {
        static let command = ["speech", "listen"]

        static let device = "--device"
        static let listDevices = "--list-devices"
        static let language = "--language"
        static let model = "--model"
        static let decodeMs = "--decode-ms"
        static let silenceMs = "--silence-ms"
        static let quiet = "--quiet"
        static let jsonl = "--jsonl"
    }
}

// MARK: - sfx generate

extension CommandFlags {
    /// `mere.run sfx generate` — Generate sound effect
    enum SFXGenerate: CommandFlagNamespace {
        static let command = ["sfx", "generate"]
        static let defaultValues = [
            "--model": "sfx-woosh-dflow",
            "--cfg": "4.5"
        ]

        static let negativePrompt = "--negative-prompt"
        static let output = "--output"
        static let model = "--model"
        static let duration = "--duration"
        static let steps = "--steps"
        static let cfg = "--cfg"
        static let seed = "--seed"
        static let renoise = "--renoise"
        static let quiet = "--quiet"
        static let progressJSON = "--progress-json"
        static let receipt = "--receipt"
    }
}

// MARK: - sfx video generate

extension CommandFlags {
    /// `mere.run sfx video generate` — Video foley
    enum SFXVideoGenerate: CommandFlagNamespace {
        static let command = ["sfx", "video", "generate"]

        static let negativePrompt = "--negative-prompt"
        static let output = "--output"
        static let model = "--model"
        static let synchformerModel = "--synchformer-model"
        static let duration = "--duration"
        static let steps = "--steps"
        static let cfg = "--cfg"
        static let seed = "--seed"
        static let renoise = "--renoise"
        static let syncBatchSize = "--sync-batch-size"
        static let clipBatchSize = "--clip-batch-size"
        static let preflight = "--preflight"
        static let json = "--json"
        static let quiet = "--quiet"
    }
}

// MARK: - sfx ae encode

extension CommandFlags {
    /// `mere.run sfx ae encode` — Encode SFX latents
    enum SFXAEEncode: CommandFlagNamespace {
        static let command = ["sfx", "ae", "encode"]

        static let output = "--output"
        static let model = "--model"
        static let quiet = "--quiet"
    }
}

// MARK: - sfx ae decode

extension CommandFlags {
    /// `mere.run sfx ae decode` — Decode SFX latents
    enum SFXAEDecode: CommandFlagNamespace {
        static let command = ["sfx", "ae", "decode"]

        static let output = "--output"
        static let model = "--model"
        static let quiet = "--quiet"
    }
}

// MARK: - sfx clap score

extension CommandFlags {
    /// `mere.run sfx clap score` — CLAP score
    enum SFXClapScore: CommandFlagNamespace {
        static let command = ["sfx", "clap", "score"]

        static let model = "--model"
        static let quiet = "--quiet"
    }
}

// MARK: - sfx condition text

extension CommandFlags {
    /// `mere.run sfx condition text` — SFX text conditioning
    enum SFXConditionText: CommandFlagNamespace {
        static let command = ["sfx", "condition", "text"]

        static let output = "--output"
        static let model = "--model"
        static let quiet = "--quiet"
    }
}

// MARK: - plugin list

extension CommandFlags {
    /// `mere.run plugin list` — List plugins
    enum PluginList: CommandFlagNamespace {
        static let command = ["plugin", "list"]

        static let catalogURL = "--catalog-url"
        static let json = "--json"
    }
}

// MARK: - plugin info

extension CommandFlags {
    /// `mere.run plugin info` — Plugin details
    enum PluginInfo: CommandFlagNamespace {
        static let command = ["plugin", "info"]

        static let catalogURL = "--catalog-url"
        static let channel = "--channel"
        static let json = "--json"
    }
}

// MARK: - plugin install

extension CommandFlags {
    /// `mere.run plugin install` — Install plugin
    enum PluginInstall: CommandFlagNamespace {
        static let command = ["plugin", "install"]

        static let catalogURL = "--catalog-url"
        static let channel = "--channel"
        static let yes = "--yes"
        static let force = "--force"
    }
}

// MARK: - plugin doctor

extension CommandFlags {
    /// `mere.run plugin doctor` — Plugin doctor
    enum PluginDoctor: CommandFlagNamespace {
        static let command = ["plugin", "doctor"]

        static let catalogURL = "--catalog-url"
    }
}

// MARK: - plugin run

extension CommandFlags {
    /// `mere.run plugin run` — Run plugin
    enum PluginRun: CommandFlagNamespace {
        static let command = ["plugin", "run"]
    }
}

// MARK: - plugin rollback

extension CommandFlags {
    /// `mere.run plugin rollback` — Roll back plugin
    enum PluginRollback: CommandFlagNamespace {
        static let command = ["plugin", "rollback"]

        static let yes = "--yes"
    }
}

// MARK: - open-webui quickstart

extension CommandFlags {
    /// `mere.run open-webui quickstart` — Open WebUI quickstart
    enum OpenWebUIQuickstart: CommandFlagNamespace {
        static let command = ["open-webui", "quickstart"]

        static let host = "--host"
        static let port = "--port"
        static let engine = "--engine"
        static let webuiHost = "--webui-host"
        static let webuiPort = "--webui-port"
        static let containerName = "--container-name"
        static let volumeName = "--volume-name"
        static let image = "--image"
        static let apiKey = "--api-key"
        static let textModel = "--text-model"
        static let visionModel = "--vision-model"
        static let embeddingModel = "--embedding-model"
        static let imageModel = "--image-model"
        static let ttsModel = "--tts-model"
        static let sttModel = "--stt-model"
        static let ttsFormat = "--tts-format"
        static let adminEmail = "--admin-email"
        static let adminPassword = "--admin-password"
        static let waitSeconds = "--wait-seconds"
        static let pull = "--pull"
        static let acceptModelLicense = "--accept-model-license"
        static let skipServer = "--skip-server"
        static let skipDocker = "--skip-docker"
        static let skipConfigure = "--skip-configure"
        static let reset = "--reset"
        static let dryRun = "--dry-run"
        static let quiet = "--quiet"
    }
}

// MARK: - api serve

extension CommandFlags {
    /// `mere.run api serve` — API server
    enum APIServe: CommandFlagNamespace {
        static let command = ["api", "serve"]

        static let port = "--port"
        static let host = "--host"
        static let model = "--model"
        static let engine = "--engine"
        static let lora = "--lora"
        static let apiKey = "--api-key"
        static let rateLimitPerMinute = "--rate-limit-per-minute"
        static let maxActiveRequests = "--max-active-requests"
        static let memoryGuard = "--memory-guard"
        static let memoryGuardCustomCeilingGb = "--memory-guard-custom-ceiling-gb"
        static let contextSize = "--context-size"
        static let kvBits = "--kv-bits"
        static let kvQuantScheme = "--kv-quant-scheme"
        static let kvGroupSize = "--kv-group-size"
        static let quantizedKVStart = "--quantized-kv-start"
        static let preflight = "--preflight"
        static let json = "--json"
    }
}

// MARK: - guide

extension CommandFlags {
    /// `mere.run guide` — Offline guides
    enum Guide: CommandFlagNamespace {
        static let command = ["guide"]

        static let list = "--list"
        static let model = "--model"
        static let json = "--json"
        static let markdown = "--markdown"
    }
}

// MARK: - config set

extension CommandFlags {
    /// `mere.run config set` — Set configuration
    enum ConfigSet: CommandFlagNamespace {
        static let command = ["config", "set"]

        static let fromEnv = "--from-env"
    }
}

// MARK: - config get

extension CommandFlags {
    /// `mere.run config get` — Read configuration
    enum ConfigGet: CommandFlagNamespace {
        static let command = ["config", "get"]

        static let reveal = "--reveal"
    }
}

// MARK: - config unset

extension CommandFlags {
    /// `mere.run config unset` — Unset configuration
    enum ConfigUnset: CommandFlagNamespace {
        static let command = ["config", "unset"]
    }
}

// MARK: - config list

extension CommandFlags {
    /// `mere.run config list` — List configuration
    enum ConfigList: CommandFlagNamespace {
        static let command = ["config", "list"]
    }
}

// MARK: - config path

extension CommandFlags {
    /// `mere.run config path` — Configuration path
    enum ConfigPath: CommandFlagNamespace {
        static let command = ["config", "path"]
    }
}

// MARK: - geo flood

extension CommandFlags {
    /// `mere.run geo flood` — Flood inference
    enum GeoFlood: CommandFlagNamespace {
        static let command = ["geo", "flood"]

        static let output = "--output"
        static let model = "--model"
        static let preflight = "--preflight"
        static let json = "--json"
    }
}

// MARK: - geo fire

extension CommandFlags {
    /// `mere.run geo fire` — Fire inference
    enum GeoFire: CommandFlagNamespace {
        static let command = ["geo", "fire"]

        static let output = "--output"
        static let model = "--model"
        static let preflight = "--preflight"
        static let json = "--json"
    }
}

// MARK: - geo tessera

extension CommandFlags {
    /// `mere.run geo tessera` — TESSERA embeddings
    enum GeoTessera: CommandFlagNamespace {
        static let command = ["geo", "tessera"]

        static let output = "--output"
        static let model = "--model"
        static let dimensions = "--dimensions"
        static let preflight = "--preflight"
        static let json = "--json"
    }
}

// MARK: - geo olmoearth

extension CommandFlags {
    /// `mere.run geo olmoearth` — OlmoEarth embeddings
    enum GeoOlmoEarth: CommandFlagNamespace {
        static let command = ["geo", "olmoearth"]

        static let output = "--output"
        static let model = "--model"
        static let patchSize = "--patch-size"
        static let inputResolution = "--input-resolution"
        static let includeTokens = "--include-tokens"
        static let preflight = "--preflight"
        static let json = "--json"
    }
}
