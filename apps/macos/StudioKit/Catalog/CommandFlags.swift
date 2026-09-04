// Generated from MereRunCapabilityCatalog by
// StudioKitTests/CommandFlagsGenerationTests.swift. Do not edit by hand:
// run ./scripts/update-studio-command-flags.sh after changing the shared contract.

/// Every flag the shared contract declares, as a constant per capability.
///
/// `CommandArguments` builds argv from these instead of string literals, so a flag the
/// CLI renames or drops is a compile error in the app rather than a command line that
/// only fails when it runs.
package enum CommandFlags {}

// MARK: - text chat

extension CommandFlags {
    /// `mere.run text chat` — Chat
    package enum TextChat: CommandFlagNamespace {
        package static let command = ["text", "chat"]
        package static let defaultValues = [
            "--max-tokens": "2048",
            "--response-format": "text",
            "--lora-scale": "1.0"
        ]

        package static let prompt = "--prompt"
        package static let image = "--image"
        package static let system = "--system"
        package static let maxTokens = "--max-tokens"
        package static let contextSize = "--context-size"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let topK = "--top-k"
        package static let minP = "--min-p"
        package static let kvBits = "--kv-bits"
        package static let kvQuantScheme = "--kv-quant-scheme"
        package static let kvGroupSize = "--kv-group-size"
        package static let quantizedKVStart = "--quantized-kv-start"
        package static let modelRoot = "--model-root"
        package static let model = "--model"
        package static let responseFormat = "--response-format"
        package static let lora = "--lora"
        package static let loraScale = "--lora-scale"
        package static let thinking = "--thinking"
        package static let noThinking = "--no-thinking"
        package static let reasoningEffort = "--reasoning-effort"
        package static let stats = "--stats"
        package static let stream = "--stream"
        package static let tools = "--tools"
        package static let toolLoop = "--tool-loop"
        package static let sandboxDir = "--sandbox-dir"
        package static let allowShellExec = "--allow-shell-exec"
        package static let allowAbsoluteToolPaths = "--allow-absolute-tool-paths"
        package static let autoApproveTools = "--auto-approve-tools"
        package static let quiet = "--quiet"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let requireInstalled = "--require-installed"
    }
}

// MARK: - text code

extension CommandFlags {
    /// `mere.run text code` — Code
    package enum TextCode: CommandFlagNamespace {
        package static let command = ["text", "code"]
        package static let defaultValues = [
            "--system": "You are a helpful coding assistant.",
            "--max-tokens": "2048",
            "--temperature": "1.0",
            "--top-p": "0.95",
            "--min-p": "0.0"
        ]

        package static let prompt = "--prompt"
        package static let system = "--system"
        package static let maxTokens = "--max-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let minP = "--min-p"
        package static let model = "--model"
        package static let stats = "--stats"
        package static let quiet = "--quiet"
        package static let stream = "--stream"
    }
}

// MARK: - text embed

extension CommandFlags {
    /// `mere.run text embed` — Embeddings
    package enum TextEmbed: CommandFlagNamespace {
        package static let command = ["text", "embed"]

        package static let model = "--model"
        package static let maxTokens = "--max-tokens"
        package static let output = "--output"
        package static let pretty = "--pretty"
    }
}

// MARK: - text anonymize

extension CommandFlags {
    /// `mere.run text anonymize` — Anonymize
    package enum TextAnonymize: CommandFlagNamespace {
        package static let command = ["text", "anonymize"]

        package static let model = "--model"
        package static let maxTokens = "--max-tokens"
        package static let replacement = "--replacement"
        package static let json = "--json"
        package static let pretty = "--pretty"
        package static let output = "--output"
    }
}

// MARK: - text train-lora

extension CommandFlags {
    /// `mere.run text train-lora` — Train text LoRA
    package enum TextTrainLoRA: CommandFlagNamespace {
        package static let command = ["text", "train-lora"]

        package static let data = "--data"
        package static let output = "--output"
        package static let model = "--model"
        package static let modelPath = "--model-path"
        package static let eval = "--eval"
        package static let adapterName = "--adapter-name"
        package static let trainingSteps = "--training-steps"
        package static let batchSize = "--batch-size"
        package static let learningRate = "--learning-rate"
        package static let rank = "--rank"
        package static let alpha = "--alpha"
        package static let maxSequenceLength = "--max-sequence-length"
        package static let reasoningEffort = "--reasoning-effort"
        package static let seed = "--seed"
        package static let targetModules = "--target-modules"
        package static let dryRun = "--dry-run"
        package static let visualize = "--visualize"
        package static let visualizePort = "--visualize-port"
        package static let json = "--json"
    }
}

// MARK: - image generate

extension CommandFlags {
    /// `mere.run image generate` — Generate and edit images
    package enum ImageGenerate: CommandFlagNamespace {
        package static let command = ["image", "generate"]
        package static let defaultValues = [
            "--width": "1024",
            "--height": "1024",
            "--mask-feather": "8",
            "--max-sequence-length": "512",
            "--structured-prompt-model": "text-chat-gemma4-12b-4bit",
            "--structured-prompt-max-tokens": "2048",
            "--lora-scale": "1.0"
        ]

        package static let prompt = "--prompt"
        package static let negativePrompt = "--negative-prompt"
        package static let cfg = "--cfg"
        package static let sigmaShift = "--sigma-shift"
        package static let output = "--output"
        package static let width = "--width"
        package static let height = "--height"
        package static let steps = "--steps"
        package static let seed = "--seed"
        package static let model = "--model"
        package static let input = "--input"
        package static let mask = "--mask"
        package static let outpaint = "--outpaint"
        package static let maskFeather = "--mask-feather"
        package static let refImage = "--ref-image"
        package static let keepOriginalAspect = "--keep-original-aspect"
        package static let strength = "--strength"
        package static let maxSequenceLength = "--max-sequence-length"
        package static let structuredPrompt = "--structured-prompt"
        package static let structuredPromptModel = "--structured-prompt-model"
        package static let structuredPromptModelRoot = "--structured-prompt-model-root"
        package static let structuredPromptMaxTokens = "--structured-prompt-max-tokens"
        package static let structuredPromptOutput = "--structured-prompt-output"
        package static let lora = "--lora"
        package static let loraScale = "--lora-scale"
        package static let kreaConditioningMultiplier = "--krea-conditioning-multiplier"
        package static let kreaConditioningLayerWeights = "--krea-conditioning-layer-weights"
        package static let kreaBaseQuantizationBits = "--krea-base-quantization-bits"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let quiet = "--quiet"
        package static let progressJSON = "--progress-json"
        package static let receipt = "--receipt"
    }
}

// MARK: - image train-lora

extension CommandFlags {
    /// `mere.run image train-lora` — Train image LoRA
    package enum ImageTrainLoRA: CommandFlagNamespace {
        package static let command = ["image", "train-lora"]

        package static let data = "--data"
        package static let output = "--output"
        package static let model = "--model"
        package static let width = "--width"
        package static let height = "--height"
        package static let trainingSteps = "--training-steps"
        package static let batchSize = "--batch-size"
        package static let learningRate = "--learning-rate"
        package static let rank = "--rank"
        package static let alpha = "--alpha"
        package static let maxTextLength = "--max-text-length"
        package static let schedulerSteps = "--scheduler-steps"
        package static let captionDropout = "--caption-dropout"
        package static let seed = "--seed"
        package static let lite = "--lite"
        package static let baseQuantizationBits = "--base-quantization-bits"
        package static let excludePreviewImages = "--exclude-preview-images"
        package static let checkpointInterval = "--checkpoint-interval"
        package static let resumeFrom = "--resume-from"
        package static let maxResolution = "--max-resolution"
        package static let progressive = "--progressive"
        package static let lowRam = "--low-ram"
        package static let noCompile = "--no-compile"
        package static let gradientCheckpointing = "--gradient-checkpointing"
        package static let recipe = "--recipe"
        package static let benchmarkSteps = "--benchmark-steps"
        package static let benchmarkWarmupSteps = "--benchmark-warmup-steps"
        package static let sampleInterval = "--sample-interval"
        package static let samplePrompt = "--sample-prompt"
        package static let sampleModel = "--sample-model"
        package static let sampleSteps = "--sample-steps"
        package static let sampleCfg = "--sample-cfg"
        package static let sampleLoRAScale = "--sample-lora-scale"
        package static let sampleSeed = "--sample-seed"
        package static let visualize = "--visualize"
        package static let visualizePort = "--visualize-port"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let loraTargetRanks = "--lora-target-ranks"
        package static let loraRankPreset = "--lora-rank-preset"
        package static let loraTargetPreset = "--lora-target-preset"
        package static let loraTargetMode = "--lora-target-mode"
        package static let timestepSampling = "--timestep-sampling"
        package static let timestepLossWeighting = "--timestep-loss-weighting"
        package static let lossWeighting = "--loss-weighting"
        package static let timestepLow = "--timestep-low"
        package static let timestepHigh = "--timestep-high"
        package static let lrWarmupSteps = "--lr-warmup-steps"
        package static let noCosineScheduler = "--no-cosine-scheduler"
        package static let lrMinFactor = "--lr-min-factor"
        package static let adamWeightDecay = "--adam-weight-decay"
        package static let syntheticSamples = "--synthetic-samples"
        package static let quiet = "--quiet"
    }
}

// MARK: - image validate

extension CommandFlags {
    /// `mere.run image validate` — Validate image runtime
    package enum ImageValidate: CommandFlagNamespace {
        package static let command = ["image", "validate"]

        package static let test = "--test"
        package static let family = "--family"
        package static let output = "--output"
        package static let saveReference = "--save-reference"
        package static let compare = "--compare"
        package static let referenceDir = "--reference-dir"
    }
}

// MARK: - image dataset discover

extension CommandFlags {
    /// `mere.run image dataset discover` — Discover image datasets
    package enum ImageDatasetDiscover: CommandFlagNamespace {
        package static let command = ["image", "dataset", "discover"]

        package static let root = "--root"
        package static let maxDepth = "--max-depth"
        package static let minUsablePairs = "--min-usable-pairs"
        package static let trainingOutputRoot = "--training-output-root"
        package static let trainingModel = "--training-model"
        package static let trainingRecipe = "--training-recipe"
        package static let excludePreviewImages = "--exclude-preview-images"
        package static let json = "--json"
    }
}

// MARK: - image run-plan

extension CommandFlags {
    /// `mere.run image run-plan` — Run image plan
    package enum ImageRunPlan: CommandFlagNamespace {
        package static let command = ["image", "run-plan"]

        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let materialize = "--materialize"
    }
}

// MARK: - image visualize-run

extension CommandFlags {
    /// `mere.run image visualize-run` — Visualize image run
    package enum ImageVisualizeRun: CommandFlagNamespace {
        package static let command = ["image", "visualize-run"]

        package static let port = "--port"
    }
}

// MARK: - image reconstruct-3d

extension CommandFlags {
    /// `mere.run image reconstruct-3d` — TripoSR reconstruction
    package enum ImageReconstruct3D: CommandFlagNamespace {
        package static let command = ["image", "reconstruct-3d"]

        package static let output = "--output"
        package static let model = "--model"
        package static let resolution = "--resolution"
        package static let densityThreshold = "--density-threshold"
        package static let foregroundRatio = "--foreground-ratio"
        package static let alreadyFramed = "--already-framed"
        package static let noVertexColors = "--no-vertex-colors"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - image reconstruct-3d-trellis2

extension CommandFlags {
    /// `mere.run image reconstruct-3d-trellis2` — TRELLIS.2 reconstruction
    package enum ImageReconstruct3DTrellis2: CommandFlagNamespace {
        package static let command = ["image", "reconstruct-3d-trellis2"]

        package static let output = "--output"
        package static let model = "--model"
        package static let seed = "--seed"
        package static let textureSeed = "--texture-seed"
        package static let maxTokens = "--max-tokens"
        package static let alreadyFramed = "--already-framed"
        package static let noRemesh = "--no-remesh"
        package static let remeshBand = "--remesh-band"
        package static let sealRadius = "--seal-radius"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - image reconstruct-3d-multiview

extension CommandFlags {
    /// `mere.run image reconstruct-3d-multiview` — InstantMesh multiview reconstruction
    package enum ImageReconstruct3DMultiview: CommandFlagNamespace {
        package static let command = ["image", "reconstruct-3d-multiview"]

        package static let view = "--view"
        package static let output = "--output"
        package static let model = "--model"
        package static let cameras = "--cameras"
        package static let resolution = "--resolution"
        package static let noVertexColors = "--no-vertex-colors"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - vision embed

extension CommandFlags {
    /// `mere.run vision embed` — Multimodal embeddings
    package enum VisionEmbed: CommandFlagNamespace {
        package static let command = ["vision", "embed"]

        package static let text = "--text"
        package static let image = "--image"
        package static let inputJSON = "--input-json"
        package static let instruction = "--instruction"
        package static let model = "--model"
        package static let dimensions = "--dimensions"
        package static let maxTokens = "--max-tokens"
        package static let minPixels = "--min-pixels"
        package static let maxPixels = "--max-pixels"
        package static let output = "--output"
        package static let pretty = "--pretty"
    }
}

// MARK: - vision inspect

extension CommandFlags {
    /// `mere.run vision inspect` — Inspect image
    package enum VisionInspect: CommandFlagNamespace {
        package static let command = ["vision", "inspect"]
        package static let defaultValues = [
            "--max-tokens": "2048",
            "--temperature": "0.7",
            "--top-p": "0.9"
        ]

        package static let prompt = "--prompt"
        package static let model = "--model"
        package static let maxTokens = "--max-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
    }
}

// MARK: - vision caption

extension CommandFlags {
    /// `mere.run vision caption` — Caption images
    package enum VisionCaption: CommandFlagNamespace {
        package static let command = ["vision", "caption"]
        package static let defaultValues = [
            "--max-tokens": "96",
            "--temperature": "0.2",
            "--top-p": "0.9"
        ]

        package static let model = "--model"
        package static let outputDir = "--output-dir"
        package static let prompt = "--prompt"
        package static let promptFile = "--prompt-file"
        package static let focus = "--focus"
        package static let triggerToken = "--trigger-token"
        package static let maxTokens = "--max-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
    }
}

// MARK: - vision ocr

extension CommandFlags {
    /// `mere.run vision ocr` — OCR
    package enum VisionOCR: CommandFlagNamespace {
        package static let command = ["vision", "ocr"]
        package static let defaultValues = [
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

        package static let backend = "--backend"
        package static let compare = "--compare"
        package static let model = "--model"
        package static let glmocrCli = "--glmocr-cli"
        package static let glmConfig = "--glm-config"
        package static let infinityRuntime = "--infinity-runtime"
        package static let infinityParserCli = "--infinity-parser-cli"
        package static let infinityModel = "--infinity-model"
        package static let infinityBackend = "--infinity-backend"
        package static let infinityAPIURL = "--infinity-api-url"
        package static let infinityAPIKey = "--infinity-api-key"
        package static let infinityTask = "--infinity-task"
        package static let infinityPrompt = "--infinity-prompt"
        package static let infinityOutputFormat = "--infinity-output-format"
        package static let infinityBatchSize = "--infinity-batch-size"
        package static let infinityModelCacheDir = "--infinity-model-cache-dir"
        package static let infinityMinPixels = "--infinity-min-pixels"
        package static let infinityMaxPixels = "--infinity-max-pixels"
        package static let outputDir = "--output-dir"
        package static let maxTokens = "--max-tokens"
        package static let temperature = "--temperature"
        package static let quiet = "--quiet"
    }
}

// MARK: - vision ground

extension CommandFlags {
    /// `mere.run vision ground` — Ground objects
    package enum VisionGround: CommandFlagNamespace {
        package static let command = ["vision", "ground"]

        package static let query = "--query"
        package static let model = "--model"
        package static let output = "--output"
        package static let jsonOutput = "--json-output"
        package static let maskOutputDir = "--mask-output-dir"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let quiet = "--quiet"
        package static let receipt = "--receipt"
    }
}

// MARK: - vision segment

extension CommandFlags {
    /// `mere.run vision segment` — Segment objects
    package enum VisionSegment: CommandFlagNamespace {
        package static let command = ["vision", "segment"]
        package static let defaultValues = [
            "--threshold": "0.05",
            "--resolution": "1008"
        ]

        package static let prompt = "--prompt"
        package static let box = "--box"
        package static let point = "--point"
        package static let model = "--model"
        package static let output = "--output"
        package static let jsonOutput = "--json-output"
        package static let maskOutputDir = "--mask-output-dir"
        package static let threshold = "--threshold"
        package static let resolution = "--resolution"
        package static let showBoxes = "--show-boxes"
        package static let multimask = "--multimask"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let quiet = "--quiet"
        package static let receipt = "--receipt"
    }
}

// MARK: - vision track

extension CommandFlags {
    /// `mere.run vision track` — Track objects
    package enum VisionTrack: CommandFlagNamespace {
        package static let command = ["vision", "track"]
        package static let defaultValues = [
            "--init-frame": "0",
            "--threshold": "0.05",
            "--resolution": "1008"
        ]

        package static let prompt = "--prompt"
        package static let box = "--box"
        package static let point = "--point"
        package static let model = "--model"
        package static let output = "--output"
        package static let jsonOutput = "--json-output"
        package static let maskOutputDir = "--mask-output-dir"
        package static let initFrame = "--init-frame"
        package static let endFrame = "--end-frame"
        package static let threshold = "--threshold"
        package static let resolution = "--resolution"
        package static let showBoxes = "--show-boxes"
        package static let showLabels = "--show-labels"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let quiet = "--quiet"
        package static let receipt = "--receipt"
    }
}

// MARK: - vision track-live

extension CommandFlags {
    /// `mere.run vision track-live` — Track camera
    package enum VisionTrackLive: CommandFlagNamespace {
        package static let command = ["vision", "track-live"]

        package static let prompt = "--prompt"
        package static let model = "--model"
        package static let output = "--output"
        package static let jsonOutput = "--json-output"
        package static let camera = "--camera"
        package static let durationSeconds = "--duration-seconds"
        package static let initFrame = "--init-frame"
        package static let seedSearchFrames = "--seed-search-frames"
        package static let threshold = "--threshold"
        package static let resolution = "--resolution"
        package static let showBoxes = "--show-boxes"
        package static let showLabels = "--show-labels"
    }
}

// MARK: - vision face detect

extension CommandFlags {
    /// `mere.run vision face detect` — Detect faces
    package enum VisionFaceDetect: CommandFlagNamespace {
        package static let command = ["vision", "face", "detect"]

        package static let model = "--model"
        package static let scoreThreshold = "--score-threshold"
        package static let executionProvider = "--execution-provider"
        package static let jsonOutput = "--json-output"
        package static let json = "--json"
        package static let maxFaces = "--max-faces"
        package static let includeEmbeddings = "--include-embeddings"
    }
}

// MARK: - vision face embed

extension CommandFlags {
    /// `mere.run vision face embed` — Embed face
    package enum VisionFaceEmbed: CommandFlagNamespace {
        package static let command = ["vision", "face", "embed"]

        package static let model = "--model"
        package static let scoreThreshold = "--score-threshold"
        package static let executionProvider = "--execution-provider"
        package static let jsonOutput = "--json-output"
        package static let json = "--json"
        package static let faceIndex = "--face-index"
    }
}

// MARK: - vision face compare

extension CommandFlags {
    /// `mere.run vision face compare` — Compare faces
    package enum VisionFaceCompare: CommandFlagNamespace {
        package static let command = ["vision", "face", "compare"]

        package static let model = "--model"
        package static let scoreThreshold = "--score-threshold"
        package static let executionProvider = "--execution-provider"
        package static let jsonOutput = "--json-output"
        package static let json = "--json"
        package static let referenceFaceIndex = "--reference-face-index"
        package static let candidateFaceIndex = "--candidate-face-index"
    }
}

// MARK: - vision face batch

extension CommandFlags {
    /// `mere.run vision face batch` — Batch faces
    package enum VisionFaceBatch: CommandFlagNamespace {
        package static let command = ["vision", "face", "batch"]

        package static let inputList = "--input-list"
        package static let model = "--model"
        package static let scoreThreshold = "--score-threshold"
        package static let executionProvider = "--execution-provider"
        package static let maxFaces = "--max-faces"
        package static let includeEmbeddings = "--include-embeddings"
        package static let jsonlOutput = "--jsonl-output"
        package static let failFast = "--fail-fast"
    }
}

// MARK: - vision pose

extension CommandFlags {
    /// `mere.run vision pose` — Pose landmarks
    package enum VisionPose: CommandFlagNamespace {
        package static let command = ["vision", "pose"]

        package static let jsonOutput = "--json-output"
        package static let noBody = "--no-body"
        package static let noHands = "--no-hands"
        package static let noFace = "--no-face"
        package static let maxHands = "--max-hands"
        package static let minimumConfidence = "--minimum-confidence"
        package static let json = "--json"
    }
}

// MARK: - vision flow

extension CommandFlags {
    /// `mere.run vision flow` — Optical flow
    package enum VisionFlow: CommandFlagNamespace {
        package static let command = ["vision", "flow"]

        package static let output = "--output"
        package static let jsonOutput = "--json-output"
        package static let accuracy = "--accuracy"
        package static let json = "--json"
    }
}

// MARK: - vision depth-video

extension CommandFlags {
    /// `mere.run vision depth-video` — Video depth
    package enum VisionDepthVideo: CommandFlagNamespace {
        package static let command = ["vision", "depth-video"]

        package static let output = "--output"
        package static let model = "--model"
        package static let inputSize = "--input-size"
        package static let maxFrames = "--max-frames"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - vision geometry

extension CommandFlags {
    /// `mere.run vision geometry` — Metric geometry
    package enum VisionGeometry: CommandFlagNamespace {
        package static let command = ["vision", "geometry"]

        package static let output = "--output"
        package static let model = "--model"
        package static let resolutionLevel = "--resolution-level"
        package static let tokenCount = "--token-count"
        package static let maxPoints = "--max-points"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - vision geometry-multiview

extension CommandFlags {
    /// `mere.run vision geometry-multiview` — Multi-view geometry
    package enum VisionGeometryMultiview: CommandFlagNamespace {
        package static let command = ["vision", "geometry-multiview"]

        package static let output = "--output"
        package static let model = "--model"
        package static let cameras = "--cameras"
        package static let processResolution = "--process-resolution"
        package static let referenceView = "--reference-view"
        package static let confidencePercentile = "--confidence-percentile"
        package static let maxPoints = "--max-points"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - audio enhance

extension CommandFlags {
    /// `mere.run audio enhance` — Enhance audio
    package enum AudioEnhance: CommandFlagNamespace {
        package static let command = ["audio", "enhance"]

        package static let model = "--model"
        package static let modelPath = "--model-path"
        package static let output = "--output"
        package static let overlap = "--overlap"
        package static let inputRate = "--input-rate"
        package static let odeMethod = "--ode-method"
        package static let odeSteps = "--ode-steps"
        package static let guidanceScale = "--guidance-scale"
        package static let seed = "--seed"
        package static let chunkSeconds = "--chunk-seconds"
        package static let dtype = "--dtype"
        package static let quiet = "--quiet"
    }
}

// MARK: - audio generate

extension CommandFlags {
    /// `mere.run audio generate` — Generate LTX audio
    package enum AudioGenerate: CommandFlagNamespace {
        package static let command = ["audio", "generate"]

        package static let output = "--output"
        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let negativePrompt = "--negative-prompt"
        package static let enhancePrompt = "--enhance-prompt"
        package static let promptEnhancerModel = "--prompt-enhancer-model"
        package static let promptEnhancerModelRoot = "--prompt-enhancer-model-root"
        package static let duration = "--duration"
        package static let autoDuration = "--auto-duration"
        package static let numFrames = "--num-frames"
        package static let fps = "--fps"
        package static let steps = "--steps"
        package static let seed = "--seed"
        package static let audioCfgGuidanceScale = "--audio-cfg-guidance-scale"
        package static let audioStgGuidanceScale = "--audio-stg-guidance-scale"
        package static let audioRescale = "--audio-rescale"
        package static let audioStgBlock = "--audio-stg-block"
        package static let audioSkipStep = "--audio-skip-step"
        package static let sigmas = "--sigmas"
        package static let lora = "--lora"
        package static let quiet = "--quiet"
    }
}

// MARK: - music generate

extension CommandFlags {
    /// `mere.run music generate` — Generate and edit music
    package enum MusicGenerate: CommandFlagNamespace {
        package static let command = ["music", "generate"]
        package static let defaultValues = [
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

        package static let lyrics = "--lyrics"
        package static let lyricsFile = "--lyrics-file"
        package static let instrumental = "--instrumental"
        package static let lrcFile = "--lrc-file"
        package static let lrcOutput = "--lrc-output"
        package static let output = "--output"
        package static let exportFormat = "--export-format"
        package static let normalize = "--normalize"
        package static let targetPeakDb = "--target-peak-db"
        package static let fadeInMs = "--fade-in-ms"
        package static let fadeOutMs = "--fade-out-ms"
        package static let noDither = "--no-dither"
        package static let recipeOutput = "--recipe-output"
        package static let noRecipe = "--no-recipe"
        package static let dawBundle = "--daw-bundle"
        package static let stems = "--stems"
        package static let adapter = "--adapter"
        package static let adapterKind = "--adapter-kind"
        package static let adapterScale = "--adapter-scale"
        package static let model = "--model"
        package static let checkpointsRoot = "--checkpoints-root"
        package static let decoderSubdirectory = "--decoder-subdirectory"
        package static let vaeSubdirectory = "--vae-subdirectory"
        package static let lmSubdirectory = "--lm-subdirectory"
        package static let lmModel = "--lm-model"
        package static let textSubdirectory = "--text-subdirectory"
        package static let useLM = "--use-lm"
        package static let noLM = "--no-lm"
        package static let analyzeSourceAudio = "--analyze-source-audio"
        package static let duration = "--duration"
        package static let quality = "--quality"
        package static let steps = "--steps"
        package static let shift = "--shift"
        package static let inferMethod = "--infer-method"
        package static let sampler = "--sampler"
        package static let guidanceScale = "--guidance-scale"
        package static let guidanceMode = "--guidance-mode"
        package static let cfgIntervalStart = "--cfg-interval-start"
        package static let cfgIntervalEnd = "--cfg-interval-end"
        package static let velocityNormThreshold = "--velocity-norm-threshold"
        package static let velocityEmaFactor = "--velocity-ema-factor"
        package static let seed = "--seed"
        package static let candidates = "--candidates"
        package static let keepCandidates = "--keep-candidates"
        package static let audioCoverStrength = "--audio-cover-strength"
        package static let coverNoiseStrength = "--cover-noise-strength"
        package static let retakeSeed = "--retake-seed"
        package static let retakeVariance = "--retake-variance"
        package static let vocalLanguage = "--vocal-language"
        package static let instruction = "--instruction"
        package static let taskType = "--task-type"
        package static let sourceAudio = "--source-audio"
        package static let referenceAudio = "--reference-audio"
        package static let trackName = "--track-name"
        package static let completeTrackClasses = "--complete-track-classes"
        package static let nonCover = "--non-cover"
        package static let repaintStart = "--repaint-start"
        package static let repaintEnd = "--repaint-end"
        package static let chunkMaskMode = "--chunk-mask-mode"
        package static let repaintMode = "--repaint-mode"
        package static let repaintStrength = "--repaint-strength"
        package static let flowEdit = "--flow-edit"
        package static let sourceCaption = "--source-caption"
        package static let sourceLyrics = "--source-lyrics"
        package static let flowEditNMin = "--flow-edit-n-min"
        package static let flowEditNMax = "--flow-edit-n-max"
        package static let flowEditNAverage = "--flow-edit-n-average"
        package static let bpm = "--bpm"
        package static let keyscale = "--keyscale"
        package static let timesignature = "--timesignature"
        package static let lmTemperature = "--lm-temperature"
        package static let lmTopK = "--lm-top-k"
        package static let lmTopP = "--lm-top-p"
        package static let lmRepetitionPenalty = "--lm-repetition-penalty"
        package static let lmCfgScale = "--lm-cfg-scale"
        package static let lmNegativePrompt = "--lm-negative-prompt"
        package static let metadataDuration = "--metadata-duration"
        package static let metadataLanguage = "--metadata-language"
        package static let noTiledVAE = "--no-tiled-vae"
        package static let vaeChunkSize = "--vae-chunk-size"
        package static let vaeOverlap = "--vae-overlap"
        package static let quiet = "--quiet"
        package static let temperature = "--temperature"
        package static let styleConditioning = "--style-conditioning"
        package static let topK = "--top-k"
        package static let cfgMusiccoca = "--cfg-musiccoca"
        package static let cfgNotes = "--cfg-notes"
        package static let cfgDrums = "--cfg-drums"
        package static let drumless = "--drumless"
        package static let unmaskWidth = "--unmask-width"
        package static let seedRotation = "--seed-rotation"
        package static let prefillSilence = "--prefill-silence"
        package static let prefillDuration = "--prefill-duration"
        package static let progressJSON = "--progress-json"
        package static let receipt = "--receipt"
    }
}

// MARK: - music analyze

extension CommandFlags {
    /// `mere.run music analyze` — Analyze music
    package enum MusicAnalyze: CommandFlagNamespace {
        package static let command = ["music", "analyze"]

        package static let model = "--model"
        package static let checkpointsRoot = "--checkpoints-root"
        package static let decoderSubdirectory = "--decoder-subdirectory"
        package static let vaeSubdirectory = "--vae-subdirectory"
        package static let lmSubdirectory = "--lm-subdirectory"
        package static let lmModel = "--lm-model"
        package static let duration = "--duration"
        package static let maxNewTokens = "--max-new-tokens"
        package static let lmTemperature = "--lm-temperature"
        package static let lmTopK = "--lm-top-k"
        package static let lmTopP = "--lm-top-p"
        package static let includeRawLM = "--include-raw-lm"
        package static let includeAudioCodes = "--include-audio-codes"
        package static let quiet = "--quiet"
    }
}

// MARK: - music transcribe

extension CommandFlags {
    /// `mere.run music transcribe` — Transcribe music
    package enum MusicTranscribe: CommandFlagNamespace {
        package static let command = ["music", "transcribe"]

        package static let model = "--model"
        package static let modelPath = "--model-path"
        package static let variant = "--variant"
        package static let output = "--output"
        package static let format = "--format"
        package static let instruments = "--instruments"
        package static let listInstruments = "--list-instruments"
        package static let sampling = "--sampling"
        package static let temperature = "--temperature"
        package static let maxTokensPerChunk = "--max-tokens-per-chunk"
        package static let strictEos = "--strict-eos"
        package static let beamSize = "--beam-size"
        package static let chunkBatchSize = "--chunk-batch-size"
        package static let dtype = "--dtype"
        package static let noMusicalContext = "--no-musical-context"
        package static let contextOutput = "--context-output"
        package static let quiet = "--quiet"
    }
}

// MARK: - music separate

extension CommandFlags {
    /// `mere.run music separate` — Separate or restore music
    package enum MusicSeparate: CommandFlagNamespace {
        package static let command = ["music", "separate"]

        package static let model = "--model"
        package static let modelPath = "--model-path"
        package static let outputDir = "--output-dir"
        package static let overlap = "--overlap"
        package static let dtype = "--dtype"
        package static let quiet = "--quiet"
    }
}

// MARK: - music realtime

extension CommandFlags {
    /// `mere.run music realtime` — Realtime music
    package enum MusicRealtime: CommandFlagNamespace {
        package static let command = ["music", "realtime"]

        package static let model = "--model"
        package static let duration = "--duration"
        package static let output = "--output"
        package static let noPlay = "--no-play"
        package static let styleConditioning = "--style-conditioning"
        package static let temperature = "--temperature"
        package static let topK = "--top-k"
        package static let cfgMusiccoca = "--cfg-musiccoca"
        package static let cfgNotes = "--cfg-notes"
        package static let cfgDrums = "--cfg-drums"
        package static let drumless = "--drumless"
        package static let unmaskWidth = "--unmask-width"
        package static let seedRotation = "--seed-rotation"
        package static let prefillSilence = "--prefill-silence"
        package static let prefillDuration = "--prefill-duration"
        package static let interactive = "--interactive"
        package static let listMidiInputs = "--list-midi-inputs"
        package static let midiMonitor = "--midi-monitor"
        package static let midiLogEvents = "--midi-log-events"
        package static let midiLogRaw = "--midi-log-raw"
        package static let midiInput = "--midi-input"
        package static let midiChannel = "--midi-channel"
        package static let midiNoteOffset = "--midi-note-offset"
        package static let midiCc = "--midi-cc"
        package static let quiet = "--quiet"
    }
}

// MARK: - music train-adapter

extension CommandFlags {
    /// `mere.run music train-adapter` — Train music adapter
    package enum MusicTrainAdapter: CommandFlagNamespace {
        package static let command = ["music", "train-adapter"]

        package static let model = "--model"
        package static let dataset = "--dataset"
        package static let output = "--output"
        package static let kind = "--kind"
        package static let rank = "--rank"
        package static let alpha = "--alpha"
        package static let factor = "--factor"
        package static let steps = "--steps"
        package static let learningRate = "--learning-rate"
        package static let weightDecay = "--weight-decay"
        package static let seed = "--seed"
        package static let maxDuration = "--max-duration"
        package static let checkpointsRoot = "--checkpoints-root"
        package static let decoderSubdirectory = "--decoder-subdirectory"
        package static let vaeSubdirectory = "--vae-subdirectory"
        package static let textSubdirectory = "--text-subdirectory"
        package static let logEvery = "--log-every"
    }
}

// MARK: - music serve

extension CommandFlags {
    /// `mere.run music serve` — Serve resident music
    package enum MusicServe: CommandFlagNamespace {
        package static let command = ["music", "serve"]

        package static let host = "--host"
        package static let port = "--port"
        package static let model = "--model"
        package static let checkpointsRoot = "--checkpoints-root"
        package static let decoderSubdirectory = "--decoder-subdirectory"
        package static let vaeSubdirectory = "--vae-subdirectory"
        package static let lmSubdirectory = "--lm-subdirectory"
        package static let lmModel = "--lm-model"
        package static let textSubdirectory = "--text-subdirectory"
        package static let adapter = "--adapter"
        package static let adapterKind = "--adapter-kind"
        package static let adapterScale = "--adapter-scale"
        package static let apiKey = "--api-key"
    }
}

// MARK: - video generate

extension CommandFlags {
    /// `mere.run video generate` — Generate video
    package enum VideoGenerate: CommandFlagNamespace {
        package static let command = ["video", "generate"]
        package static let defaultValues = [
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

        package static let output = "--output"
        package static let model = "--model"
        package static let quality = "--quality"
        package static let outputMode = "--output-mode"
        package static let modelRoot = "--model-root"
        package static let autoDuration = "--auto-duration"
        package static let videoDecoder = "--video-decoder"
        package static let hdr = "--hdr"
        package static let hdrTransfer = "--hdr-transfer"
        package static let highQualityHdr = "--high-quality-hdr"
        package static let textEmbeddings = "--text-embeddings"
        package static let spatialTile = "--spatial-tile"
        package static let spatialOverlap = "--spatial-overlap"
        package static let skipMp4 = "--skip-mp4"
        package static let width = "--width"
        package static let height = "--height"
        package static let numFrames = "--num-frames"
        package static let duration = "--duration"
        package static let fps = "--fps"
        package static let seed = "--seed"
        package static let steps = "--steps"
        package static let h3WeightMode = "--h3-weight-mode"
        package static let h3Acceleration = "--h3-acceleration"
        package static let guidanceScale = "--guidance-scale"
        package static let shift = "--shift"
        package static let negativePrompt = "--negative-prompt"
        package static let enhancePrompt = "--enhance-prompt"
        package static let promptEnhancerModel = "--prompt-enhancer-model"
        package static let promptEnhancerModelRoot = "--prompt-enhancer-model-root"
        package static let audio = "--audio"
        package static let audioStartTime = "--audio-start-time"
        package static let audioMaxDuration = "--audio-max-duration"
        package static let a2vGuidanceScale = "--a2v-guidance-scale"
        package static let videoCfgGuidanceScale = "--video-cfg-guidance-scale"
        package static let audioCfgGuidanceScale = "--audio-cfg-guidance-scale"
        package static let v2aGuidanceScale = "--v2a-guidance-scale"
        package static let a2vSteps = "--a2v-steps"
        package static let ltxPreset = "--ltx-preset"
        package static let ltxPipeline = "--ltx-pipeline"
        package static let ltxSampler = "--ltx-sampler"
        package static let ltxSigmas = "--ltx-sigmas"
        package static let ltxStage2Sigmas = "--ltx-stage-2-sigmas"
        package static let distilledLoRAStrengthStage1 = "--distilled-lora-strength-stage-1"
        package static let distilledLoRAStrengthStage2 = "--distilled-lora-strength-stage-2"
        package static let ltxSamplerEta = "--ltx-sampler-eta"
        package static let videoStgScale = "--video-stg-scale"
        package static let videoGuidanceRescale = "--video-guidance-rescale"
        package static let videoStgBlock = "--video-stg-block"
        package static let videoGuidanceSkipStep = "--video-guidance-skip-step"
        package static let audioStgScale = "--audio-stg-scale"
        package static let audioGuidanceRescale = "--audio-guidance-rescale"
        package static let audioStgBlock = "--audio-stg-block"
        package static let audioGuidanceSkipStep = "--audio-guidance-skip-step"
        package static let noRes2sBongMath = "--no-res2s-bong-math"
        package static let res2sBongMaxIterations = "--res2s-bong-max-iterations"
        package static let gradientEstimationGamma = "--gradient-estimation-gamma"
        package static let image = "--image"
        package static let imageStrength = "--image-strength"
        package static let endImage = "--end-image"
        package static let endImageStrength = "--end-image-strength"
        package static let imageConditioning = "--image-conditioning"
        package static let numGeneratedKeyframes = "--num-generated-keyframes"
        package static let generatedKeyframe = "--generated-keyframe"
        package static let lora = "--lora"
        package static let videoConditioning = "--video-conditioning"
        package static let conditioningAttentionStrength = "--conditioning-attention-strength"
        package static let conditioningAttentionMask = "--conditioning-attention-mask"
        package static let skipStage2 = "--skip-stage-2"
        package static let referenceDownscaleFactor = "--reference-downscale-factor"
        package static let referenceTemporalScaleFactor = "--reference-temporal-scale-factor"
        package static let dfr = "--dfr"
        package static let temporalUpsampleRounds = "--temporal-upsample-rounds"
        package static let detailingLoRA = "--detailing-lora"
        package static let detailingReferenceDownscaleFactor = "--detailing-reference-downscale-factor"
        package static let reference = "--reference"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let timings = "--timings"
        package static let timingsOutput = "--timings-output"
        package static let quiet = "--quiet"
        package static let progressJSON = "--progress-json"
        package static let receipt = "--receipt"
    }
}

// MARK: - video retake

extension CommandFlags {
    /// `mere.run video retake` — Retake video region
    package enum VideoRetake: CommandFlagNamespace {
        package static let command = ["video", "retake"]

        package static let source = "--source"
        package static let frameRate = "--frame-rate"
        package static let startTime = "--start-time"
        package static let endTime = "--end-time"
        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let output = "--output"
        package static let seed = "--seed"
        package static let negativePrompt = "--negative-prompt"
        package static let enhancePrompt = "--enhance-prompt"
        package static let promptEnhancerModel = "--prompt-enhancer-model"
        package static let promptEnhancerModelRoot = "--prompt-enhancer-model-root"
        package static let steps = "--steps"
        package static let sigmas = "--sigmas"
        package static let lora = "--lora"
        package static let videoCfgGuidanceScale = "--video-cfg-guidance-scale"
        package static let videoStgScale = "--video-stg-scale"
        package static let videoGuidanceRescale = "--video-guidance-rescale"
        package static let videoModalityScale = "--video-modality-scale"
        package static let videoStgBlock = "--video-stg-block"
        package static let videoGuidanceSkipStep = "--video-guidance-skip-step"
        package static let audioCfgGuidanceScale = "--audio-cfg-guidance-scale"
        package static let audioStgScale = "--audio-stg-scale"
        package static let audioGuidanceRescale = "--audio-guidance-rescale"
        package static let audioModalityScale = "--audio-modality-scale"
        package static let audioStgBlock = "--audio-stg-block"
        package static let audioGuidanceSkipStep = "--audio-guidance-skip-step"
        package static let videoDecoder = "--video-decoder"
        package static let hdr = "--hdr"
        package static let hdrTransfer = "--hdr-transfer"
        package static let preserveVideo = "--preserve-video"
        package static let preserveAudio = "--preserve-audio"
        package static let quiet = "--quiet"
    }
}

// MARK: - video dub-it

extension CommandFlags {
    /// `mere.run video dub-it` — Dub-It
    package enum VideoDubIt: CommandFlagNamespace {
        package static let command = ["video", "dub-it"]

        package static let referenceVideo = "--reference-video"
        package static let icLoRA = "--ic-lora"
        package static let icLoRAStrength = "--ic-lora-strength"
        package static let referenceStrength = "--reference-strength"
        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let output = "--output"
        package static let width = "--width"
        package static let height = "--height"
        package static let seed = "--seed"
        package static let imageConditioning = "--image-conditioning"
        package static let stage1Sigmas = "--stage-1-sigmas"
        package static let stage2Sigmas = "--stage-2-sigmas"
        package static let enhancePrompt = "--enhance-prompt"
        package static let promptEnhancerModel = "--prompt-enhancer-model"
        package static let promptEnhancerModelRoot = "--prompt-enhancer-model-root"
        package static let videoDecoder = "--video-decoder"
        package static let quiet = "--quiet"
    }
}

// MARK: - video animate

extension CommandFlags {
    /// `mere.run video animate` — Animate subject
    package enum VideoAnimate: CommandFlagNamespace {
        package static let command = ["video", "animate"]

        package static let reference = "--reference"
        package static let referenceMask = "--reference-mask"
        package static let drivingVideo = "--driving-video"
        package static let drivingMask = "--driving-mask"
        package static let additionalReference = "--additional-reference"
        package static let additionalReferenceMask = "--additional-reference-mask"
        package static let output = "--output"
        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let mode = "--mode"
        package static let profile = "--profile"
        package static let width = "--width"
        package static let height = "--height"
        package static let steps = "--steps"
        package static let guidanceScale = "--guidance-scale"
        package static let shift = "--shift"
        package static let sampler = "--sampler"
        package static let distilledAdapter = "--distilled-adapter"
        package static let distilledAdapterStrength = "--distilled-adapter-strength"
        package static let seed = "--seed"
        package static let fps = "--fps"
        package static let segmentLength = "--segment-length"
        package static let segmentOverlap = "--segment-overlap"
        package static let tailPolicy = "--tail-policy"
        package static let audioSource = "--audio-source"
        package static let negativePrompt = "--negative-prompt"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let quiet = "--quiet"
    }
}

// MARK: - video cosmos3

extension CommandFlags {
    /// `mere.run video cosmos3` — Cosmos3
    package enum VideoCosmos3: CommandFlagNamespace {
        package static let command = ["video", "cosmos3"]

        package static let mode = "--mode"
        package static let model = "--model"
        package static let output = "--output"
        package static let actionsOutput = "--actions-output"
        package static let image = "--image"
        package static let video = "--video"
        package static let negativePrompt = "--negative-prompt"
        package static let width = "--width"
        package static let height = "--height"
        package static let numFrames = "--num-frames"
        package static let steps = "--steps"
        package static let guidanceScale = "--guidance-scale"
        package static let shift = "--shift"
        package static let schedule = "--schedule"
        package static let seed = "--seed"
        package static let fps = "--fps"
        package static let conditionLatentFrame = "--condition-latent-frame"
        package static let keepVideoTail = "--keep-video-tail"
        package static let actionDomain = "--action-domain"
        package static let actionFile = "--action-file"
        package static let actionChunkSize = "--action-chunk-size"
        package static let actionResolution = "--action-resolution"
        package static let actionViewpoint = "--action-viewpoint"
        package static let maxNewTokens = "--max-new-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let maxVideoFrames = "--max-video-frames"
        package static let quiet = "--quiet"
    }
}

// MARK: - video prepare-masks

extension CommandFlags {
    /// `mere.run video prepare-masks` — Prepare SCAIL-2 masks
    package enum VideoPrepareMasks: CommandFlagNamespace {
        package static let command = ["video", "prepare-masks"]

        package static let plan = "--plan"
        package static let outputDir = "--output-dir"
        package static let previewFrame = "--preview-frame"
        package static let model = "--model"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let quiet = "--quiet"
    }
}

// MARK: - video export-latents

extension CommandFlags {
    /// `mere.run video export-latents` — Export video latents
    package enum VideoExportLatents: CommandFlagNamespace {
        package static let command = ["video", "export-latents"]

        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let output = "--output"
        package static let width = "--width"
        package static let height = "--height"
        package static let numFrames = "--num-frames"
        package static let seed = "--seed"
        package static let quiet = "--quiet"
    }
}

// MARK: - video session

extension CommandFlags {
    /// `mere.run video session` — Resident LTX session
    package enum VideoSession: CommandFlagNamespace {
        package static let command = ["video", "session"]

        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let quiet = "--quiet"
    }
}

// MARK: - adapter list

extension CommandFlags {
    /// `mere.run adapter list` — Browse adapters
    package enum AdapterList: CommandFlagNamespace {
        package static let command = ["adapter", "list"]

        package static let json = "--json"
    }
}

// MARK: - adapter pull

extension CommandFlags {
    /// `mere.run adapter pull` — Pull adapter
    package enum AdapterPull: CommandFlagNamespace {
        package static let command = ["adapter", "pull"]

        package static let force = "--force"
        package static let quiet = "--quiet"
    }
}

// MARK: - run list

extension CommandFlags {
    /// `mere.run run list` — Browse durable runs
    package enum RunList: CommandFlagNamespace {
        package static let command = ["run", "list"]

        package static let root = "--root"
        package static let executor = "--executor"
        package static let limit = "--limit"
        package static let maxDepth = "--max-depth"
        package static let json = "--json"
    }
}

// MARK: - run inspect

extension CommandFlags {
    /// `mere.run run inspect` — Inspect durable run
    package enum RunInspect: CommandFlagNamespace {
        package static let command = ["run", "inspect"]

        package static let json = "--json"
    }
}

// MARK: - run watch

extension CommandFlags {
    /// `mere.run run watch` — Watch remote run
    package enum RunWatch: CommandFlagNamespace {
        package static let command = ["run", "watch"]

        package static let pollInterval = "--poll-interval"
        package static let jsonStream = "--json-stream"
        package static let json = "--json"
    }
}

// MARK: - run fetch

extension CommandFlags {
    /// `mere.run run fetch` — Fetch remote run
    package enum RunFetch: CommandFlagNamespace {
        package static let command = ["run", "fetch"]

        package static let into = "--into"
        package static let allArtifacts = "--all-artifacts"
        package static let artifact = "--artifact"
        package static let json = "--json"
    }
}

// MARK: - run cancel

extension CommandFlags {
    /// `mere.run run cancel` — Cancel run
    package enum RunCancel: CommandFlagNamespace {
        package static let command = ["run", "cancel"]

        package static let json = "--json"
    }
}

// MARK: - run retry

extension CommandFlags {
    /// `mere.run run retry` — Retry Relay run
    package enum RunRetry: CommandFlagNamespace {
        package static let command = ["run", "retry"]

        package static let json = "--json"
    }
}

// MARK: - eval pack validate

extension CommandFlags {
    /// `mere.run eval pack validate` — Validate evaluation pack
    package enum EvalPackValidate: CommandFlagNamespace {
        package static let command = ["eval", "pack", "validate"]

        package static let json = "--json"
    }
}

// MARK: - eval run

extension CommandFlags {
    /// `mere.run eval run` — Run external evaluation
    package enum EvalRun: CommandFlagNamespace {
        package static let command = ["eval", "run"]

        package static let model = "--model"
        package static let adapter = "--adapter"
        package static let trials = "--trials"
        package static let maxTokens = "--max-tokens"
        package static let contextSize = "--context-size"
        package static let logprobs = "--logprobs"
        package static let topLogprobs = "--top-logprobs"
        package static let allowExternalScorer = "--allow-external-scorer"
        package static let logResponses = "--log-responses"
        package static let dryRun = "--dry-run"
        package static let checkpoint = "--checkpoint"
        package static let resume = "--resume"
        package static let caseTrialLimit = "--case-trial-limit"
        package static let output = "--output"
        package static let json = "--json"
    }
}

// MARK: - eval promote

extension CommandFlags {
    /// `mere.run eval promote` — Promote evaluated artifact
    package enum EvalPromote: CommandFlagNamespace {
        package static let command = ["eval", "promote"]

        package static let output = "--output"
        package static let json = "--json"
    }
}

// MARK: - world serve

extension CommandFlags {
    /// `mere.run world serve` — World session
    package enum WorldServe: CommandFlagNamespace {
        package static let command = ["world", "serve"]

        package static let host = "--host"
        package static let port = "--port"
        package static let apiKey = "--api-key"
        package static let backend = "--backend"
        package static let baseModel = "--base-model"
        package static let model = "--model"
        package static let stateDirectory = "--state-directory"
        package static let prepare = "--prepare"
    }
}

// MARK: - vision serve

extension CommandFlags {
    /// `mere.run vision serve` — Vision grounding server
    package enum VisionServe: CommandFlagNamespace {
        package static let command = ["vision", "serve"]

        package static let host = "--host"
        package static let port = "--port"
        package static let model = "--model"
        package static let apiKey = "--api-key"
        package static let maxFrameBytes = "--max-frame-bytes"
        package static let maxBatchSize = "--max-batch-size"
        package static let maxBatchBytes = "--max-batch-bytes"
        package static let preflight = "--preflight"
        package static let json = "--json"
    }
}

// MARK: - status

extension CommandFlags {
    /// `mere.run status` — Status snapshot
    package enum Status: CommandFlagNamespace {
        package static let command = ["status"]

        package static let host = "--host"
        package static let port = "--port"
        package static let apiKey = "--api-key"
        package static let timeoutSeconds = "--timeout-seconds"
        package static let json = "--json"
    }
}

// MARK: - gate

extension CommandFlags {
    /// `mere.run gate` — Quality gate
    package enum Gate: CommandFlagNamespace {
        package static let command = ["gate"]

        package static let suite = "--suite"
        package static let updateBaselines = "--update-baselines"
        package static let strictPerf = "--strict-perf"
        package static let jsonOutput = "--json-output"
        package static let list = "--list"
    }
}

// MARK: - model storage

extension CommandFlags {
    /// `mere.run model storage` — Model storage
    package enum ModelStorage: CommandFlagNamespace {
        package static let command = ["model", "storage"]

        package static let json = "--json"
    }
}

// MARK: - model location list

extension CommandFlags {
    /// `mere.run model location list` — List model locations
    package enum ModelLocationList: CommandFlagNamespace {
        package static let command = ["model", "location", "list"]

        package static let json = "--json"
    }
}

// MARK: - model location add

extension CommandFlags {
    /// `mere.run model location add` — Add search root
    package enum ModelLocationAdd: CommandFlagNamespace {
        package static let command = ["model", "location", "add"]
    }
}

// MARK: - model location remove

extension CommandFlags {
    /// `mere.run model location remove` — Remove search root
    package enum ModelLocationRemove: CommandFlagNamespace {
        package static let command = ["model", "location", "remove"]
    }
}

// MARK: - model location bind

extension CommandFlags {
    /// `mere.run model location bind` — Bind model directory
    package enum ModelLocationBind: CommandFlagNamespace {
        package static let command = ["model", "location", "bind"]

        package static let acceptModelLicense = "--accept-model-license"
    }
}

// MARK: - model location unbind

extension CommandFlags {
    /// `mere.run model location unbind` — Unbind model directory
    package enum ModelLocationUnbind: CommandFlagNamespace {
        package static let command = ["model", "location", "unbind"]
    }
}

// MARK: - model gc

extension CommandFlags {
    /// `mere.run model gc` — Model storage cleanup
    package enum ModelGc: CommandFlagNamespace {
        package static let command = ["model", "gc"]

        package static let force = "--force"
        package static let json = "--json"
    }
}

// MARK: - model runtime get

extension CommandFlags {
    /// `mere.run model runtime get` — Read runtime policy
    package enum ModelRuntimeGet: CommandFlagNamespace {
        package static let command = ["model", "runtime", "get"]

        package static let json = "--json"
    }
}

// MARK: - model runtime set

extension CommandFlags {
    /// `mere.run model runtime set` — Set runtime policy
    package enum ModelRuntimeSet: CommandFlagNamespace {
        package static let command = ["model", "runtime", "set"]

        package static let alias = "--alias"
        package static let clearAlias = "--clear-alias"
        package static let pinned = "--pinned"
        package static let unpinned = "--unpinned"
        package static let ttlSeconds = "--ttl-seconds"
        package static let clearTTL = "--clear-ttl"
        package static let maxContextTokens = "--max-context-tokens"
        package static let clearMaxContextTokens = "--clear-max-context-tokens"
        package static let maxTokens = "--max-tokens"
        package static let clearMaxTokens = "--clear-max-tokens"
        package static let temperature = "--temperature"
        package static let clearTemperature = "--clear-temperature"
        package static let topP = "--top-p"
        package static let clearTopP = "--clear-top-p"
        package static let minP = "--min-p"
        package static let clearMinP = "--clear-min-p"
        package static let engine = "--engine"
        package static let clearEngine = "--clear-engine"
        package static let kvCacheMode = "--kv-cache-mode"
        package static let clearKVCacheMode = "--clear-kv-cache-mode"
        package static let json = "--json"
    }
}

// MARK: - setup

extension CommandFlags {
    /// `mere.run setup` — Setup path
    package enum Setup: CommandFlagNamespace {
        package static let command = ["setup"]

        package static let mode = "--mode"
        package static let agentModel = "--agent-model"
        package static let install = "--install"
        package static let start = "--start"
        package static let dryRun = "--dry-run"
        package static let host = "--host"
        package static let port = "--port"
        package static let piPath = "--pi-path"
        package static let quiet = "--quiet"
    }
}

// MARK: - agent onboard

extension CommandFlags {
    /// `mere.run agent onboard` — Agent onboarding
    package enum AgentOnboard: CommandFlagNamespace {
        package static let command = ["agent", "onboard"]

        package static let pullRecommended = "--pull-recommended"
        package static let acceptModelLicense = "--accept-model-license"
        package static let installPi = "--install-pi"
        package static let configurePi = "--configure-pi"
        package static let host = "--host"
        package static let port = "--port"
        package static let model = "--model"
        package static let quiet = "--quiet"
    }
}

// MARK: - agent status

extension CommandFlags {
    /// `mere.run agent status` — Agent status
    package enum AgentStatus: CommandFlagNamespace {
        package static let command = ["agent", "status"]

        package static let piPath = "--pi-path"
        package static let json = "--json"
    }
}

// MARK: - agent install-pi

extension CommandFlags {
    /// `mere.run agent install-pi` — Install Pi
    package enum AgentInstallPi: CommandFlagNamespace {
        package static let command = ["agent", "install-pi"]

        package static let force = "--force"
    }
}

// MARK: - agent start

extension CommandFlags {
    /// `mere.run agent start` — Start setup agent
    package enum AgentStart: CommandFlagNamespace {
        package static let command = ["agent", "start"]

        package static let host = "--host"
        package static let port = "--port"
        package static let piPath = "--pi-path"
        package static let prompt = "--prompt"
        package static let model = "--model"
        package static let skipServer = "--skip-server"
        package static let allowUnsupported = "--allow-unsupported"
        package static let noBootstrap = "--no-bootstrap"
        package static let quiet = "--quiet"
    }
}

// MARK: - model list

extension CommandFlags {
    /// `mere.run model list` — List models
    package enum ModelList: CommandFlagNamespace {
        package static let command = ["model", "list"]

        package static let measureSizes = "--measure-sizes"
        package static let json = "--json"
    }
}

// MARK: - model capabilities

extension CommandFlags {
    /// `mere.run model capabilities` — Model capabilities
    package enum ModelCapabilities: CommandFlagNamespace {
        package static let command = ["model", "capabilities"]

        package static let all = "--all"
        package static let recommended = "--recommended"
        package static let json = "--json"
    }
}

// MARK: - model pull

extension CommandFlags {
    /// `mere.run model pull` — Pull model
    package enum ModelPull: CommandFlagNamespace {
        package static let command = ["model", "pull"]

        package static let all = "--all"
        package static let force = "--force"
        package static let quiet = "--quiet"
        package static let allowUnsupported = "--allow-unsupported"
        package static let acceptModelLicense = "--accept-model-license"
        package static let preflight = "--preflight"
        package static let json = "--json"
    }
}

// MARK: - model info

extension CommandFlags {
    /// `mere.run model info` — Model info
    package enum ModelInfo: CommandFlagNamespace {
        package static let command = ["model", "info"]

        package static let json = "--json"
        package static let components = "--components"
    }
}

// MARK: - model remove

extension CommandFlags {
    /// `mere.run model remove` — Remove model
    package enum ModelRemove: CommandFlagNamespace {
        package static let command = ["model", "remove"]

        package static let force = "--force"
        package static let keepCache = "--keep-cache"
        package static let json = "--json"
    }
}

// MARK: - model repair-manifests

extension CommandFlags {
    /// `mere.run model repair-manifests` — Repair manifests
    package enum ModelRepairManifests: CommandFlagNamespace {
        package static let command = ["model", "repair-manifests"]

        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - model optimize

extension CommandFlags {
    /// `mere.run model optimize` — Optimize model
    package enum ModelOptimize: CommandFlagNamespace {
        package static let command = ["model", "optimize"]

        package static let force = "--force"
        package static let output = "--output"
        package static let json = "--json"
    }
}

// MARK: - model benchmark q36-mtp

extension CommandFlags {
    /// `mere.run model benchmark q36-mtp` — Qwen-family MTP benchmark
    package enum ModelBenchmarkQ36MTP: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "q36-mtp"]

        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let prompt = "--prompt"
        package static let promptFile = "--prompt-file"
        package static let promptRepeat = "--prompt-repeat"
        package static let promptRepeatValues = "--prompt-repeat-values"
        package static let decodeTokens = "--decode-tokens"
        package static let decodeTokenValues = "--decode-token-values"
        package static let temperature = "--temperature"
        package static let temperatureValues = "--temperature-values"
        package static let topP = "--top-p"
        package static let contextSize = "--context-size"
        package static let mtpBlockSize = "--mtp-block-size"
        package static let forcedMTPMinPromptTokens = "--forced-mtp-min-prompt-tokens"
        package static let json = "--json"
    }
}

// MARK: - model benchmark laguna-dflash

extension CommandFlags {
    /// `mere.run model benchmark laguna-dflash` — Laguna DFlash benchmark
    package enum ModelBenchmarkLagunaDFlash: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "laguna-dflash"]

        package static let lagunaPath = "--laguna-path"
        package static let lagunaDFlashPath = "--laguna-dflash-path"
        package static let decodeTokenValues = "--decode-token-values"
        package static let repetitions = "--repetitions"
        package static let lagunaDFlashTokens = "--laguna-dflash-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let topK = "--top-k"
        package static let minP = "--min-p"
        package static let prompt = "--prompt"
        package static let promptFile = "--prompt-file"
        package static let fixture = "--fixture"
        package static let contextSize = "--context-size"
        package static let concurrencyValues = "--concurrency-values"
        package static let warmupRepetitions = "--warmup-repetitions"
        package static let mixedFixtures = "--mixed-fixtures"
        package static let includeAutomatic = "--include-automatic"
        package static let logResponses = "--log-responses"
        package static let json = "--json"
    }
}

// MARK: - model benchmark chat

extension CommandFlags {
    /// `mere.run model benchmark chat` — Chat benchmark
    package enum ModelBenchmarkChat: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "chat"]

        package static let models = "--models"
        package static let suite = "--suite"
        package static let cases = "--cases"
        package static let maxTokens = "--max-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let topK = "--top-k"
        package static let minP = "--min-p"
        package static let contextSize = "--context-size"
        package static let concurrency = "--concurrency"
        package static let lagunaPath = "--laguna-path"
        package static let lagunaDFlashPath = "--laguna-dflash-path"
        package static let lagunaDFlashTokens = "--laguna-dflash-tokens"
        package static let lagunaDFlashMinTokens = "--laguna-dflash-min-tokens"
        package static let lagunaDFlashRouting = "--laguna-dflash-routing"
        package static let dryRun = "--dry-run"
        package static let logResponses = "--log-responses"
        package static let json = "--json"
    }
}

// MARK: - model benchmark code

extension CommandFlags {
    /// `mere.run model benchmark code` — Code benchmark
    package enum ModelBenchmarkCode: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "code"]

        package static let models = "--models"
        package static let lagunaPath = "--laguna-path"
        package static let lagunaDFlashPath = "--laguna-dflash-path"
        package static let lagunaDFlashTokens = "--laguna-dflash-tokens"
        package static let lagunaDFlashMinTokens = "--laguna-dflash-min-tokens"
        package static let suite = "--suite"
        package static let tasks = "--tasks"
        package static let humanevalFile = "--humaneval-file"
        package static let maxTokens = "--max-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let topK = "--top-k"
        package static let minP = "--min-p"
        package static let thinking = "--thinking"
        package static let executionTimeout = "--execution-timeout"
        package static let python = "--python"
        package static let sandbox = "--sandbox"
        package static let allowCodeExecution = "--allow-code-execution"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - model benchmark fused

extension CommandFlags {
    /// `mere.run model benchmark fused` — Fused quality suite
    package enum ModelBenchmarkFused: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "fused"]

        package static let suite = "--suite"
        package static let models = "--models"
        package static let manifest = "--manifest"
        package static let externalCases = "--external-cases"
        package static let cases = "--cases"
        package static let capabilities = "--capabilities"
        package static let trials = "--trials"
        package static let maxTokens = "--max-tokens"
        package static let contextSize = "--context-size"
        package static let logprobs = "--logprobs"
        package static let topLogprobs = "--top-logprobs"
        package static let performanceLane = "--performance-lane"
        package static let executionTimeout = "--execution-timeout"
        package static let python = "--python"
        package static let sandbox = "--sandbox"
        package static let allowCodeExecution = "--allow-code-execution"
        package static let logResponses = "--log-responses"
        package static let checkpoint = "--checkpoint"
        package static let resume = "--resume"
        package static let caseTrialLimit = "--case-trial-limit"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - model benchmark fused-fixture

extension CommandFlags {
    /// `mere.run model benchmark fused-fixture` — Fused fixture hashes
    package enum ModelBenchmarkFusedFixture: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "fused-fixture"]

        package static let check = "--check"
    }
}

// MARK: - model benchmark vlm

extension CommandFlags {
    /// `mere.run model benchmark vlm` — Vision-language benchmark
    package enum ModelBenchmarkVLM: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "vlm"]

        package static let models = "--models"
        package static let dataset = "--dataset"
        package static let lmmsTasks = "--lmms-tasks"
        package static let fixtureDir = "--fixture-dir"
        package static let outputDir = "--output-dir"
        package static let lmmsEvalRoot = "--lmms-eval-root"
        package static let lmmsEvalPython = "--lmms-eval-python"
        package static let externalEndpoint = "--external-endpoint"
        package static let baseURL = "--base-url"
        package static let apiKey = "--api-key"
        package static let host = "--host"
        package static let port = "--port"
        package static let limit = "--limit"
        package static let logSamples = "--log-samples"
        package static let maxTokens = "--max-tokens"
        package static let contextSize = "--context-size"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - model benchmark tool-calls

extension CommandFlags {
    /// `mere.run model benchmark tool-calls` — Tool-call benchmark
    package enum ModelBenchmarkToolCalls: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "tool-calls"]

        package static let models = "--models"
        package static let cases = "--cases"
        package static let maxTokens = "--max-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let topK = "--top-k"
        package static let minP = "--min-p"
        package static let contextSize = "--context-size"
        package static let lagunaPath = "--laguna-path"
        package static let lagunaDFlashPath = "--laguna-dflash-path"
        package static let lagunaDFlashTokens = "--laguna-dflash-tokens"
        package static let lagunaDFlashMinTokens = "--laguna-dflash-min-tokens"
        package static let dryRun = "--dry-run"
        package static let logResponses = "--log-responses"
        package static let json = "--json"
    }
}

// MARK: - model benchmark tool-continuations

extension CommandFlags {
    /// `mere.run model benchmark tool-continuations` — Tool continuation benchmark
    package enum ModelBenchmarkToolContinuations: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "tool-continuations"]

        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let maxTokens = "--max-tokens"
        package static let contextSize = "--context-size"
        package static let dryRun = "--dry-run"
        package static let logResponses = "--log-responses"
        package static let json = "--json"
    }
}

// MARK: - model benchmark gemma4-kv

extension CommandFlags {
    /// `mere.run model benchmark gemma4-kv` — Gemma4 KV benchmark
    package enum ModelBenchmarkGemma4KV: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "gemma4-kv"]

        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let prompt = "--prompt"
        package static let promptFile = "--prompt-file"
        package static let promptRepeat = "--prompt-repeat"
        package static let promptRepeatValues = "--prompt-repeat-values"
        package static let decodeTokens = "--decode-tokens"
        package static let decodeTokenValues = "--decode-token-values"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let json = "--json"
    }
}

// MARK: - model benchmark gemma4-mtp

extension CommandFlags {
    /// `mere.run model benchmark gemma4-mtp` — Gemma4 MTP benchmark
    package enum ModelBenchmarkGemma4MTP: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "gemma4-mtp"]

        package static let model = "--model"
        package static let modelRoot = "--model-root"
        package static let prompt = "--prompt"
        package static let promptFile = "--prompt-file"
        package static let promptRepeat = "--prompt-repeat"
        package static let promptRepeatValues = "--prompt-repeat-values"
        package static let decodeTokens = "--decode-tokens"
        package static let decodeTokenValues = "--decode-token-values"
        package static let mtpBlockSize = "--mtp-block-size"
        package static let mtpMinPromptTokens = "--mtp-min-prompt-tokens"
        package static let json = "--json"
    }
}

// MARK: - model benchmark api-workload

extension CommandFlags {
    /// `mere.run model benchmark api-workload` — API workload benchmark
    package enum ModelBenchmarkAPIWorkload: CommandFlagNamespace {
        package static let command = ["model", "benchmark", "api-workload"]

        package static let host = "--host"
        package static let port = "--port"
        package static let apiKey = "--api-key"
        package static let model = "--model"
        package static let workloadFile = "--workload-file"
        package static let turns = "--turns"
        package static let sharedPrefixRepeat = "--shared-prefix-repeat"
        package static let maxTokens = "--max-tokens"
        package static let temperature = "--temperature"
        package static let topP = "--top-p"
        package static let concurrency = "--concurrency"
        package static let timeoutSeconds = "--timeout-seconds"
        package static let dryRun = "--dry-run"
        package static let json = "--json"
    }
}

// MARK: - speech synthesize

extension CommandFlags {
    /// `mere.run speech synthesize` — Synthesize speech
    package enum SpeechSynthesize: CommandFlagNamespace {
        package static let command = ["speech", "synthesize"]
        package static let defaultValues = [
            "--model": "speech-tts-qwen3-nano",
            "--voice": "A calm female voice with clear pronunciation",
            "--mode": "style",
            "--language": "auto",
            "--temperature": "0.6",
            "--stream-chunk-tokens": "25"
        ]

        package static let output = "--output"
        package static let model = "--model"
        package static let voice = "--voice"
        package static let mode = "--mode"
        package static let profile = "--profile"
        package static let refAudio = "--ref-audio"
        package static let refText = "--ref-text"
        package static let language = "--language"
        package static let saveProfile = "--save-profile"
        package static let temperature = "--temperature"
        package static let stream = "--stream"
        package static let streamChunkTokens = "--stream-chunk-tokens"
        package static let quiet = "--quiet"
        package static let progressJSON = "--progress-json"
        package static let receipt = "--receipt"
    }
}

// MARK: - speech transcribe

extension CommandFlags {
    /// `mere.run speech transcribe` — Transcribe speech
    package enum SpeechTranscribe: CommandFlagNamespace {
        package static let command = ["speech", "transcribe"]
        package static let defaultValues = [
            "--backend": "auto",
            "--task": "transcribe",
            "--max-tokens": "448",
            "--stream-chunk-ms": "200",
            "--stream-decode-ms": "2000"
        ]

        package static let output = "--output"
        package static let model = "--model"
        package static let backend = "--backend"
        package static let task = "--task"
        package static let language = "--language"
        package static let maxTokens = "--max-tokens"
        package static let stream = "--stream"
        package static let streamChunkMs = "--stream-chunk-ms"
        package static let streamDecodeMs = "--stream-decode-ms"
        package static let inputFormat = "--input-format"
        package static let sampleRate = "--sample-rate"
        package static let jsonl = "--jsonl"
        package static let noTimestamps = "--no-timestamps"
        package static let quiet = "--quiet"
        package static let receipt = "--receipt"
    }
}

// MARK: - speech diarize

extension CommandFlags {
    /// `mere.run speech diarize` — Diarize speech
    package enum SpeechDiarize: CommandFlagNamespace {
        package static let command = ["speech", "diarize"]

        package static let model = "--model"
        package static let format = "--format"
        package static let output = "--output"
        package static let threshold = "--threshold"
        package static let minDuration = "--min-duration"
        package static let mergeGap = "--merge-gap"
        package static let quiet = "--quiet"
    }
}

// MARK: - speech profile list

extension CommandFlags {
    /// `mere.run speech profile list` — Voice profiles
    package enum SpeechProfileList: CommandFlagNamespace {
        package static let command = ["speech", "profile", "list"]
    }
}

// MARK: - speech profile create

extension CommandFlags {
    /// `mere.run speech profile create` — Create voice profile
    package enum SpeechProfileCreate: CommandFlagNamespace {
        package static let command = ["speech", "profile", "create"]

        package static let name = "--name"
        package static let audio = "--audio"
        package static let text = "--text"
        package static let language = "--language"
        package static let quiet = "--quiet"
    }
}

// MARK: - speech profile delete

extension CommandFlags {
    /// `mere.run speech profile delete` — Delete voice profile
    package enum SpeechProfileDelete: CommandFlagNamespace {
        package static let command = ["speech", "profile", "delete"]

        package static let id = "--id"
    }
}

// MARK: - speech listen

extension CommandFlags {
    /// `mere.run speech listen` — Live transcription
    package enum SpeechListen: CommandFlagNamespace {
        package static let command = ["speech", "listen"]

        package static let device = "--device"
        package static let listDevices = "--list-devices"
        package static let language = "--language"
        package static let model = "--model"
        package static let decodeMs = "--decode-ms"
        package static let silenceMs = "--silence-ms"
        package static let quiet = "--quiet"
        package static let jsonl = "--jsonl"
    }
}

// MARK: - sfx generate

extension CommandFlags {
    /// `mere.run sfx generate` — Generate sound effect
    package enum SFXGenerate: CommandFlagNamespace {
        package static let command = ["sfx", "generate"]
        package static let defaultValues = [
            "--model": "sfx-woosh-dflow",
            "--cfg": "4.5"
        ]

        package static let negativePrompt = "--negative-prompt"
        package static let output = "--output"
        package static let model = "--model"
        package static let duration = "--duration"
        package static let steps = "--steps"
        package static let cfg = "--cfg"
        package static let seed = "--seed"
        package static let renoise = "--renoise"
        package static let quiet = "--quiet"
        package static let progressJSON = "--progress-json"
        package static let receipt = "--receipt"
    }
}

// MARK: - sfx video generate

extension CommandFlags {
    /// `mere.run sfx video generate` — Video foley
    package enum SFXVideoGenerate: CommandFlagNamespace {
        package static let command = ["sfx", "video", "generate"]

        package static let negativePrompt = "--negative-prompt"
        package static let output = "--output"
        package static let model = "--model"
        package static let synchformerModel = "--synchformer-model"
        package static let duration = "--duration"
        package static let steps = "--steps"
        package static let cfg = "--cfg"
        package static let seed = "--seed"
        package static let renoise = "--renoise"
        package static let syncBatchSize = "--sync-batch-size"
        package static let clipBatchSize = "--clip-batch-size"
        package static let preflight = "--preflight"
        package static let json = "--json"
        package static let quiet = "--quiet"
    }
}

// MARK: - sfx ae encode

extension CommandFlags {
    /// `mere.run sfx ae encode` — Encode SFX latents
    package enum SFXAEEncode: CommandFlagNamespace {
        package static let command = ["sfx", "ae", "encode"]

        package static let output = "--output"
        package static let model = "--model"
        package static let quiet = "--quiet"
    }
}

// MARK: - sfx ae decode

extension CommandFlags {
    /// `mere.run sfx ae decode` — Decode SFX latents
    package enum SFXAEDecode: CommandFlagNamespace {
        package static let command = ["sfx", "ae", "decode"]

        package static let output = "--output"
        package static let model = "--model"
        package static let quiet = "--quiet"
    }
}

// MARK: - sfx clap score

extension CommandFlags {
    /// `mere.run sfx clap score` — CLAP score
    package enum SFXClapScore: CommandFlagNamespace {
        package static let command = ["sfx", "clap", "score"]

        package static let model = "--model"
        package static let quiet = "--quiet"
    }
}

// MARK: - sfx condition text

extension CommandFlags {
    /// `mere.run sfx condition text` — SFX text conditioning
    package enum SFXConditionText: CommandFlagNamespace {
        package static let command = ["sfx", "condition", "text"]

        package static let output = "--output"
        package static let model = "--model"
        package static let quiet = "--quiet"
    }
}

// MARK: - plugin list

extension CommandFlags {
    /// `mere.run plugin list` — List plugins
    package enum PluginList: CommandFlagNamespace {
        package static let command = ["plugin", "list"]

        package static let catalogURL = "--catalog-url"
        package static let json = "--json"
    }
}

// MARK: - plugin info

extension CommandFlags {
    /// `mere.run plugin info` — Plugin details
    package enum PluginInfo: CommandFlagNamespace {
        package static let command = ["plugin", "info"]

        package static let catalogURL = "--catalog-url"
        package static let channel = "--channel"
        package static let json = "--json"
    }
}

// MARK: - plugin install

extension CommandFlags {
    /// `mere.run plugin install` — Install plugin
    package enum PluginInstall: CommandFlagNamespace {
        package static let command = ["plugin", "install"]

        package static let catalogURL = "--catalog-url"
        package static let channel = "--channel"
        package static let yes = "--yes"
        package static let force = "--force"
    }
}

// MARK: - plugin doctor

extension CommandFlags {
    /// `mere.run plugin doctor` — Plugin doctor
    package enum PluginDoctor: CommandFlagNamespace {
        package static let command = ["plugin", "doctor"]

        package static let catalogURL = "--catalog-url"
    }
}

// MARK: - plugin run

extension CommandFlags {
    /// `mere.run plugin run` — Run plugin
    package enum PluginRun: CommandFlagNamespace {
        package static let command = ["plugin", "run"]
    }
}

// MARK: - plugin rollback

extension CommandFlags {
    /// `mere.run plugin rollback` — Roll back plugin
    package enum PluginRollback: CommandFlagNamespace {
        package static let command = ["plugin", "rollback"]

        package static let yes = "--yes"
    }
}

// MARK: - open-webui quickstart

extension CommandFlags {
    /// `mere.run open-webui quickstart` — Open WebUI quickstart
    package enum OpenWebUIQuickstart: CommandFlagNamespace {
        package static let command = ["open-webui", "quickstart"]

        package static let host = "--host"
        package static let port = "--port"
        package static let engine = "--engine"
        package static let webuiHost = "--webui-host"
        package static let webuiPort = "--webui-port"
        package static let containerName = "--container-name"
        package static let volumeName = "--volume-name"
        package static let image = "--image"
        package static let apiKey = "--api-key"
        package static let textModel = "--text-model"
        package static let visionModel = "--vision-model"
        package static let embeddingModel = "--embedding-model"
        package static let imageModel = "--image-model"
        package static let ttsModel = "--tts-model"
        package static let sttModel = "--stt-model"
        package static let ttsFormat = "--tts-format"
        package static let adminEmail = "--admin-email"
        package static let adminPassword = "--admin-password"
        package static let waitSeconds = "--wait-seconds"
        package static let pull = "--pull"
        package static let acceptModelLicense = "--accept-model-license"
        package static let skipServer = "--skip-server"
        package static let skipDocker = "--skip-docker"
        package static let skipConfigure = "--skip-configure"
        package static let reset = "--reset"
        package static let dryRun = "--dry-run"
        package static let quiet = "--quiet"
    }
}

// MARK: - api serve

extension CommandFlags {
    /// `mere.run api serve` — API server
    package enum APIServe: CommandFlagNamespace {
        package static let command = ["api", "serve"]

        package static let port = "--port"
        package static let host = "--host"
        package static let model = "--model"
        package static let engine = "--engine"
        package static let lora = "--lora"
        package static let apiKey = "--api-key"
        package static let rateLimitPerMinute = "--rate-limit-per-minute"
        package static let maxActiveRequests = "--max-active-requests"
        package static let memoryGuard = "--memory-guard"
        package static let memoryGuardCustomCeilingGb = "--memory-guard-custom-ceiling-gb"
        package static let contextSize = "--context-size"
        package static let kvBits = "--kv-bits"
        package static let kvQuantScheme = "--kv-quant-scheme"
        package static let kvGroupSize = "--kv-group-size"
        package static let quantizedKVStart = "--quantized-kv-start"
        package static let preflight = "--preflight"
        package static let json = "--json"
    }
}

// MARK: - guide

extension CommandFlags {
    /// `mere.run guide` — Offline guides
    package enum Guide: CommandFlagNamespace {
        package static let command = ["guide"]

        package static let list = "--list"
        package static let model = "--model"
        package static let json = "--json"
        package static let markdown = "--markdown"
    }
}

// MARK: - config set

extension CommandFlags {
    /// `mere.run config set` — Set configuration
    package enum ConfigSet: CommandFlagNamespace {
        package static let command = ["config", "set"]

        package static let fromEnv = "--from-env"
    }
}

// MARK: - config get

extension CommandFlags {
    /// `mere.run config get` — Read configuration
    package enum ConfigGet: CommandFlagNamespace {
        package static let command = ["config", "get"]

        package static let reveal = "--reveal"
    }
}

// MARK: - config unset

extension CommandFlags {
    /// `mere.run config unset` — Unset configuration
    package enum ConfigUnset: CommandFlagNamespace {
        package static let command = ["config", "unset"]
    }
}

// MARK: - config list

extension CommandFlags {
    /// `mere.run config list` — List configuration
    package enum ConfigList: CommandFlagNamespace {
        package static let command = ["config", "list"]
    }
}

// MARK: - config path

extension CommandFlags {
    /// `mere.run config path` — Configuration path
    package enum ConfigPath: CommandFlagNamespace {
        package static let command = ["config", "path"]
    }
}

// MARK: - geo flood

extension CommandFlags {
    /// `mere.run geo flood` — Flood inference
    package enum GeoFlood: CommandFlagNamespace {
        package static let command = ["geo", "flood"]

        package static let output = "--output"
        package static let model = "--model"
        package static let preflight = "--preflight"
        package static let json = "--json"
    }
}

// MARK: - geo fire

extension CommandFlags {
    /// `mere.run geo fire` — Fire inference
    package enum GeoFire: CommandFlagNamespace {
        package static let command = ["geo", "fire"]

        package static let output = "--output"
        package static let model = "--model"
        package static let preflight = "--preflight"
        package static let json = "--json"
    }
}

// MARK: - geo tessera

extension CommandFlags {
    /// `mere.run geo tessera` — TESSERA embeddings
    package enum GeoTessera: CommandFlagNamespace {
        package static let command = ["geo", "tessera"]

        package static let output = "--output"
        package static let model = "--model"
        package static let dimensions = "--dimensions"
        package static let preflight = "--preflight"
        package static let json = "--json"
    }
}

// MARK: - geo olmoearth

extension CommandFlags {
    /// `mere.run geo olmoearth` — OlmoEarth embeddings
    package enum GeoOlmoEarth: CommandFlagNamespace {
        package static let command = ["geo", "olmoearth"]

        package static let output = "--output"
        package static let model = "--model"
        package static let patchSize = "--patch-size"
        package static let inputResolution = "--input-resolution"
        package static let includeTokens = "--include-tokens"
        package static let preflight = "--preflight"
        package static let json = "--json"
    }
}
