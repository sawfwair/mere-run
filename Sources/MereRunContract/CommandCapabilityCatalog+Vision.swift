import Foundation

extension MereRunCapabilityCatalog {
    public static let visionInspect = MereRunCommandCapability(
        id: "vision.inspect",
        command: ["vision", "inspect"],
        title: "Inspect image",
        summary: "Describe or answer questions about an image with a local VLM.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true),
            .init(name: "prompt-words", label: "Prompt words", kind: .string, required: false, repeatable: true)],
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, group: Group.prompt, tier: .essential),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "2048", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 8_192, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.7", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                defaultValue: "0.9", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            )
        ],
        output: .init(kind: .text)
    )

    public static let visionEmbed = MereRunCommandCapability(
        id: "vision.embed",
        command: ["vision", "embed"],
        title: "Multimodal embeddings",
        summary: "Generate shared text and image embeddings with native Qwen3-VL.",
        options: [
            .init(flag: "--text", label: "Text inputs", kind: .string, repeatable: true),
            .init(flag: "--image", label: "Image inputs", kind: .file, repeatable: true),
            .init(flag: "--input-json", label: "JSON batch", kind: .file),
            .init(flag: "--instruction", label: "Retrieval instruction", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--dimensions", label: "Dimensions", kind: .integer),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--min-pixels", label: "Minimum pixels", kind: .integer),
            .init(flag: "--max-pixels", label: "Maximum pixels", kind: .integer),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
    )

    public static let visionCaption = MereRunCommandCapability(
        id: "vision.caption",
        command: ["vision", "caption"],
        title: "Caption images",
        summary: "Generate training-friendly captions for one or more images.",
        arguments: [.init(name: "images", label: "Images", kind: .file, required: true, repeatable: true)],
        options: [
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory, group: Group.output, tier: .standard),
            .init(flag: "--prompt", label: "Prompt", kind: .string, group: Group.prompt, tier: .essential),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file, group: Group.prompt, tier: .expert),
            .init(flag: "--focus", label: "Focus", kind: .string, repeatable: true, group: Group.prompt, tier: .standard),
            .init(flag: "--trigger-token", label: "Trigger token", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "96", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 2_048, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.2", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                defaultValue: "0.9", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            )
        ],
        output: .init(kind: .directory, flag: "--output-dir")
    )

    public static let visionOCR = MereRunCommandCapability(
        id: "vision.ocr",
        command: ["vision", "ocr"],
        title: "OCR",
        summary: "Extract text with native LightOn/Infinity or external GLM/Infinity runtimes.",
        arguments: [.init(name: "images", label: "Images", kind: .file, required: true, repeatable: true)],
        options: [
            .init(
                flag: "--backend", label: "Backend", kind: .choice, choices: ["lighton", "glm", "infinity"],
                defaultValue: "lighton", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(flag: "--compare", label: "Compare", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--model", label: "LightOn model", kind: .string,
                defaultValue: "vision-ocr-lighton", group: Group.modelAndAdapters, tier: .standard
            ),
            .init(
                flag: "--glmocr-cli", label: "GLM executable", kind: .file,
                defaultValue: "glmocr", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(flag: "--glm-config", label: "GLM config", kind: .file, group: Group.modelAndAdapters, tier: .expert),
            .init(
                flag: "--infinity-runtime", label: "Infinity runtime", kind: .choice, choices: ["native", "external"],
                defaultValue: "native", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-parser-cli", label: "Parser executable", kind: .file,
                defaultValue: "parser", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-model", label: "Infinity model", kind: .string,
                defaultValue: "vision-ocr-infinity-pro-int8", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-backend",
                label: "Infinity backend",
                kind: .choice,
                choices: ["transformers", "vllm-engine", "vllm-server"],
                defaultValue: "vllm-server", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-api-url", label: "Infinity API URL", kind: .string,
                defaultValue: "http://localhost:8000/v1/chat/completions", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-api-key", label: "Infinity API key", kind: .string,
                defaultValue: "EMPTY", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-task", label: "Infinity task", kind: .choice, choices: ["doc2json", "doc2md", "custom"],
                defaultValue: "doc2json", group: Group.prompt, tier: .expert
            ),
            .init(flag: "--infinity-prompt", label: "Infinity prompt", kind: .string, group: Group.prompt, tier: .expert),
            .init(
                flag: "--infinity-output-format", label: "Infinity format", kind: .choice, choices: ["md", "json"],
                defaultValue: "md", group: Group.output, tier: .expert
            ),
            .init(
                flag: "--infinity-batch-size", label: "Batch size", kind: .integer,
                defaultValue: "1", group: Group.run, tier: .expert, range: .init(min: 1, max: 64, step: 1)
            ),
            .init(
                flag: "--infinity-model-cache-dir", label: "Model cache", kind: .directory,
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-min-pixels", label: "Minimum pixels", kind: .integer,
                defaultValue: "2048", group: Group.inputs, tier: .expert, range: .init(min: 1, step: 1)
            ),
            .init(
                flag: "--infinity-max-pixels", label: "Maximum pixels", kind: .integer,
                defaultValue: "16777216", group: Group.inputs, tier: .expert, range: .init(min: 1, step: 1)
            ),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory, group: Group.output, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "4096", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 16_384, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.2", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .text, flag: "--output-dir", optional: true)
    )

    public static let visionGround = MereRunCommandCapability(
        id: "vision.ground",
        command: ["vision", "ground"],
        title: "Ground objects",
        summary: "Ground one or more text expressions with native Falcon Perception.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--query", label: "Query", kind: .string, repeatable: true, group: Group.prompt, tier: .essential),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--output", label: "Annotated image", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--json-output", label: "JSON output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--mask-output-dir", label: "Mask directory", kind: .directory, group: Group.output, tier: .standard),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "png", flag: "--output")
    )

    public static let visionSegment = MereRunCommandCapability(
        id: "vision.segment",
        command: ["vision", "segment"],
        title: "Segment objects",
        summary: "Segment text, box, or point prompts with native SAM 3.1.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--prompt", label: "Text prompt", kind: .string, repeatable: true, group: Group.prompt, tier: .essential),
            .init(flag: "--box", label: "Box prompt", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--point", label: "Point prompt", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--output", label: "Annotated image", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--json-output", label: "JSON output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--mask-output-dir", label: "Mask directory", kind: .directory, group: Group.output, tier: .standard),
            .init(
                flag: "--threshold", label: "Threshold", kind: .number,
                defaultValue: "0.05", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--resolution", label: "Resolution", kind: .integer,
                defaultValue: "1008", group: Group.sampling, tier: .expert, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(flag: "--show-boxes", label: "Show boxes", kind: .boolean, group: Group.output, tier: .standard),
            .init(flag: "--multimask", label: "Multiple masks", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "png", flag: "--output")
    )

    public static let visionTrack = MereRunCommandCapability(
        id: "vision.track",
        command: ["vision", "track"],
        title: "Track objects",
        summary: "Track text, box, or point prompted objects through video.",
        arguments: [.init(name: "video", label: "Video", kind: .file, required: true)],
        options: [
            .init(flag: "--prompt", label: "Text prompt", kind: .string, repeatable: true, group: Group.prompt, tier: .essential),
            .init(flag: "--box", label: "Box prompt", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--point", label: "Point prompt", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--output", label: "Annotated video", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--json-output", label: "JSON output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--mask-output-dir", label: "Mask directory", kind: .directory, group: Group.output, tier: .standard),
            .init(
                flag: "--init-frame", label: "Initial frame", kind: .integer,
                defaultValue: "0", group: Group.inputs, tier: .standard, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--end-frame", label: "End frame", kind: .integer,
                group: Group.inputs, tier: .standard, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--threshold", label: "Threshold", kind: .number,
                defaultValue: "0.05", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--resolution", label: "Resolution", kind: .integer,
                defaultValue: "1008", group: Group.sampling, tier: .expert, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(flag: "--show-boxes", label: "Show boxes", kind: .boolean, group: Group.output, tier: .standard),
            .init(flag: "--show-labels", label: "Show labels", kind: .boolean, group: Group.output, tier: .expert),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let visionTrackLive = MereRunCommandCapability(
        id: "vision.track-live",
        command: ["vision", "track-live"],
        title: "Track camera",
        summary: "Capture a camera and track prompted objects.",
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, repeatable: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--output", label: "Annotated video", kind: .file, required: true),
            .init(flag: "--json-output", label: "JSON output", kind: .file),
            .init(flag: "--camera", label: "Camera", kind: .integer),
            .init(flag: "--duration-seconds", label: "Duration", kind: .number),
            .init(flag: "--init-frame", label: "Initial frame", kind: .integer),
            .init(flag: "--seed-search-frames", label: "Seed search", kind: .integer),
            .init(flag: "--threshold", label: "Threshold", kind: .number),
            .init(flag: "--resolution", label: "Resolution", kind: .integer),
            .init(flag: "--show-boxes", label: "Show boxes", kind: .boolean),
            .init(flag: "--show-labels", label: "Show labels", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let visionFaceDetect = MereRunCommandCapability(
        id: "vision.face.detect",
        command: ["vision", "face", "detect"],
        title: "Detect faces",
        summary: "Detect faces, landmarks, and optional identity embeddings.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: faceOptions + [
            .init(flag: "--max-faces", label: "Max faces", kind: .integer),
            .init(flag: "--include-embeddings", label: "Embeddings", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--json-output", optional: true)
    )

    public static let visionFaceEmbed = MereRunCommandCapability(
        id: "vision.face.embed",
        command: ["vision", "face", "embed"],
        title: "Embed face",
        summary: "Create one normalized ArcFace identity embedding.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: faceOptions + [
            .init(flag: "--face-index", label: "Face index", kind: .integer)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--json-output", optional: true)
    )

    public static let visionFaceCompare = MereRunCommandCapability(
        id: "vision.face.compare",
        command: ["vision", "face", "compare"],
        title: "Compare faces",
        summary: "Compare a face from each of two images.",
        arguments: [
            .init(name: "reference", label: "Reference", kind: .file, required: true),
            .init(name: "candidate", label: "Candidate", kind: .file, required: true)
        ],
        options: faceOptions + [
            .init(flag: "--reference-face-index", label: "Reference index", kind: .integer),
            .init(flag: "--candidate-face-index", label: "Candidate index", kind: .integer)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--json-output", optional: true)
    )

    public static let visionFaceBatch = MereRunCommandCapability(
        id: "vision.face.batch",
        command: ["vision", "face", "batch"],
        title: "Batch faces",
        summary: "Analyze many images in one warm face session.",
        arguments: [.init(name: "images", label: "Images", kind: .file, required: false, repeatable: true)],
        options: [
            .init(flag: "--input-list", label: "Input list", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--score-threshold", label: "Score threshold", kind: .number),
            .init(flag: "--execution-provider", label: "Provider", kind: .choice, choices: ["auto", "coreml", "cpu"]),
            .init(flag: "--max-faces", label: "Max faces", kind: .integer),
            .init(flag: "--include-embeddings", label: "Embeddings", kind: .boolean),
            .init(flag: "--jsonl-output", label: "JSONL output", kind: .file),
            .init(flag: "--fail-fast", label: "Fail fast", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "jsonl", flag: "--jsonl-output", optional: true)
    )

    private static let faceOptions: [MereRunCapabilityOption] = [
        .init(flag: "--model", label: "Model", kind: .string),
        .init(flag: "--score-threshold", label: "Score threshold", kind: .number),
        .init(flag: "--execution-provider", label: "Provider", kind: .choice, choices: ["auto", "coreml", "cpu"]),
        .init(flag: "--json-output", label: "JSON output", kind: .file),
        .init(flag: "--json", label: "Print JSON", kind: .boolean)
    ]

    public static let visionPose = MereRunCommandCapability(
        id: "vision.pose",
        command: ["vision", "pose"],
        title: "Pose landmarks",
        summary: "Detect body, hand, and face landmarks with native platform APIs.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--json-output", label: "JSON output", kind: .file),
            .init(flag: "--no-body", label: "Disable body", kind: .boolean),
            .init(flag: "--no-hands", label: "Disable hands", kind: .boolean),
            .init(flag: "--no-face", label: "Disable face", kind: .boolean),
            .init(flag: "--max-hands", label: "Max hands", kind: .integer),
            .init(flag: "--minimum-confidence", label: "Confidence", kind: .number),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "json", flag: "--json-output")
    )

    public static let visionFlow = MereRunCommandCapability(
        id: "vision.flow",
        command: ["vision", "flow"],
        title: "Optical flow",
        summary: "Generate dense optical flow between two images.",
        arguments: [
            .init(name: "from", label: "From", kind: .file, required: true),
            .init(name: "to", label: "To", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Flow output", kind: .file),
            .init(flag: "--json-output", label: "JSON output", kind: .file),
            .init(flag: "--accuracy", label: "Accuracy", kind: .choice, choices: ["low", "medium", "high", "very-high"]),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "flo", flag: "--output")
    )

    public static let visionDepthVideo = MereRunCommandCapability(
        id: "vision.depth-video",
        command: ["vision", "depth-video"],
        title: "Video depth",
        summary: "Generate temporally consistent relative or metric depth.",
        arguments: [.init(name: "input", label: "Video", kind: .file, required: true)],
        options: [
            .init(flag: "--output", label: "Output directory", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--input-size", label: "Input edge", kind: .integer),
            .init(flag: "--max-frames", label: "Max frames", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let visionGeometry = MereRunCommandCapability(
        id: "vision.geometry",
        command: ["vision", "geometry"],
        title: "Metric geometry",
        summary: "Generate metric depth, normals, camera intrinsics, and a point cloud.",
        arguments: [.init(name: "input", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--output", label: "Output directory", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--resolution-level", label: "Quality", kind: .integer),
            .init(flag: "--token-count", label: "Token count", kind: .integer),
            .init(flag: "--max-points", label: "Max points", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let visionGeometryMultiview = MereRunCommandCapability(
        id: "vision.geometry-multiview",
        command: ["vision", "geometry-multiview"],
        title: "Multi-view geometry",
        summary: "Solve relative geometry, confidence, and cameras from ordered views.",
        arguments: [.init(name: "images", label: "Images", kind: .file, required: true, repeatable: true)],
        options: [
            .init(flag: "--output", label: "Output directory", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--cameras", label: "Cameras", kind: .file),
            .init(flag: "--process-resolution", label: "Process resolution", kind: .integer),
            .init(flag: "--reference-view", label: "Reference view", kind: .choice, choices: ["first", "middle", "saddle-balanced", "saddle-similarity-range"]),
            .init(flag: "--confidence-percentile", label: "Confidence percentile", kind: .number),
            .init(flag: "--max-points", label: "Max points", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let visionServe = MereRunCommandCapability(
        id: "vision.serve",
        command: ["vision", "serve"],
        title: "Vision grounding server",
        summary: "Serve resident, binary-frame vision grounding over HTTP.",
        options: [
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--max-frame-bytes", label: "Max frame bytes", kind: .integer),
            .init(flag: "--max-batch-size", label: "Max batch size", kind: .integer),
            .init(flag: "--max-batch-bytes", label: "Max batch bytes", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .service)
    )
}
