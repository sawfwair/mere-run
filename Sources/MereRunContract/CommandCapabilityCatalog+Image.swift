import Foundation

extension MereRunCapabilityCatalog {
    public static let imageGenerate = MereRunCommandCapability(
        id: "image.generate",
        command: ["image", "generate"],
        title: "Generate and edit images",
        summary: "Generate, transform, or personalize images with references, structured prompts, and LoRAs.",
        options: [
            .init(flag: "--run-dir", label: "Run directory", kind: .directory, group: Group.output, tier: .expert),
            .init(
                flag: "--sigmas", label: "Sigma schedule", kind: .string, group: Group.sampling, tier: .expert
            ),
            .init(flag: "--prompt", label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--cfg", label: "CFG scale", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 20, step: 0.5)
            ),
            .init(
                flag: "--sigma-shift", label: "Sigma shift", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 16, step: 0.1)
            ),
            .init(flag: "--output", label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--width", label: "Width", kind: .integer,
                defaultValue: "1024", group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--height", label: "Height", kind: .integer,
                defaultValue: "1024", group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--steps", label: "Steps", kind: .integer,
                group: Group.sampling, tier: .essential, range: .init(min: 1, max: 100, step: 1)
            ),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--input", label: "Input image", kind: .file, group: Group.inputs, tier: .standard),
            .init(flag: "--mask", label: "Edit mask", kind: .file, group: Group.inputs, tier: .standard, dependsOn: "--input"),
            .init(
                flag: "--outpaint", label: "Outpaint padding", kind: .string,
                group: Group.inputs, tier: .expert, dependsOn: "--input"
            ),
            .init(
                flag: "--mask-feather", label: "Mask feather", kind: .integer,
                defaultValue: "8", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 128, step: 1), dependsOn: "--input"
            ),
            .init(
                flag: "--ref-image", label: "Reference image", kind: .file, repeatable: true,
                group: Group.inputs, tier: .standard
            ),
            .init(
                flag: "--keep-original-aspect", label: "Keep original aspect", kind: .boolean,
                group: Group.inputs, tier: .expert, dependsOn: "--ref-image"
            ),
            .init(
                flag: "--strength", label: "Edit strength", kind: .number,
                group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--max-sequence-length", label: "Max sequence length", kind: .integer,
                defaultValue: "512", group: Group.sampling, tier: .expert, range: .init(min: 64, max: 4_096, step: 64)
            ),
            .init(flag: "--structured-prompt", label: "Structured prompt", kind: .boolean, group: Group.prompt, tier: .standard),
            .init(
                flag: "--structured-prompt-model", label: "Prompt model", kind: .string,
                defaultValue: "text-chat-gemma4-12b-4bit", group: Group.prompt, tier: .expert, dependsOn: "--structured-prompt"
            ),
            .init(
                flag: "--structured-prompt-model-root", label: "Prompt model root", kind: .directory,
                group: Group.prompt, tier: .expert, dependsOn: "--structured-prompt"
            ),
            .init(
                flag: "--structured-prompt-max-tokens", label: "Prompt max tokens", kind: .integer,
                defaultValue: "2048", group: Group.prompt, tier: .expert,
                range: .init(min: 1, max: 8_192, step: 1), dependsOn: "--structured-prompt"
            ),
            .init(
                flag: "--structured-prompt-output", label: "Structured prompt output", kind: .file,
                group: Group.prompt, tier: .expert, dependsOn: "--structured-prompt"
            ),
            .init(flag: "--lora", label: "LoRA", kind: .file, repeatable: true, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--lora-scale", label: "LoRA scale", kind: .number,
                defaultValue: "1.0", group: Group.modelAndAdapters, tier: .standard,
                range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--lora"
            ),
            .init(
                flag: "--krea-conditioning-multiplier", label: "Krea conditioning", kind: .number,
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--krea-conditioning-layer-weights", label: "Krea layer weights", kind: .string,
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--krea-base-quantization-bits",
                label: "Krea base quantization",
                kind: .choice,
                choices: ["4", "8"],
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "png", flag: "--output")
    )

    public static let imageTrainLoRA = MereRunCommandCapability(
        id: "image.train-lora",
        command: ["image", "train-lora"],
        title: "Train image LoRA",
        summary: "Train Krea 2 or FLUX.2 Klein adapters with recipes, previews, checkpoints, and dashboards.",
        options: [
            .init(flag: "--data", label: "Dataset", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file, required: true),
            .init(flag: "--model", label: "Base model", kind: .string),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--training-steps", label: "Training steps", kind: .integer),
            .init(flag: "--batch-size", label: "Batch size", kind: .integer),
            .init(flag: "--learning-rate", label: "Learning rate", kind: .number),
            .init(flag: "--rank", label: "Rank", kind: .integer),
            .init(flag: "--alpha", label: "Alpha", kind: .number),
            .init(flag: "--max-text-length", label: "Max text length", kind: .integer),
            .init(flag: "--scheduler-steps", label: "Scheduler steps", kind: .integer),
            .init(flag: "--caption-dropout", label: "Caption dropout", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--lite", label: "Lite targets", kind: .boolean),
            .init(flag: "--base-quantization-bits", label: "Base quantization", kind: .choice, choices: ["4", "8"]),
            .init(flag: "--exclude-preview-images", label: "Exclude preview images", kind: .boolean),
            .init(flag: "--checkpoint-interval", label: "Checkpoint interval", kind: .integer),
            .init(flag: "--resume-from", label: "Resume checkpoint", kind: .file),
            .init(flag: "--max-resolution", label: "Max resolution", kind: .integer),
            .init(flag: "--progressive", label: "Progressive resolution", kind: .boolean),
            .init(flag: "--low-ram", label: "Low RAM", kind: .boolean),
            .init(flag: "--no-compile", label: "Disable compile", kind: .boolean),
            .init(flag: "--gradient-checkpointing", label: "Gradient checkpointing", kind: .boolean),
            .init(
                flag: "--recipe",
                label: "Recipe",
                kind: .choice,
                choices: ["krea-fast-style", "krea-cinematic-style", "klein-fast-style"]
            ),
            .init(flag: "--benchmark-steps", label: "Benchmark steps", kind: .integer),
            .init(flag: "--benchmark-warmup-steps", label: "Benchmark warmup", kind: .integer),
            .init(flag: "--sample-interval", label: "Sample interval", kind: .integer),
            .init(flag: "--sample-prompt", label: "Sample prompt", kind: .string),
            .init(flag: "--sample-model", label: "Sample model", kind: .string),
            .init(flag: "--sample-steps", label: "Sample steps", kind: .integer),
            .init(flag: "--sample-cfg", label: "Sample CFG", kind: .number),
            .init(flag: "--sample-lora-scale", label: "Sample LoRA scale", kind: .number),
            .init(flag: "--sample-seed", label: "Sample seed", kind: .integer),
            .init(flag: "--visualize", label: "Visualize", kind: .boolean),
            .init(flag: "--visualize-port", label: "Visualization port", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--lora-target-ranks", label: "Target ranks", kind: .string),
            .init(flag: "--lora-rank-preset", label: "Rank preset", kind: .choice, choices: ["flux2-style-128"]),
            .init(flag: "--lora-target-preset", label: "Target preset", kind: .choice, choices: ["fal-klein-fast"]),
            .init(
                flag: "--lora-target-mode",
                label: "Target mode",
                kind: .choice,
                choices: ["suffix", "transformer-linear-walk"]
            ),
            .init(
                flag: "--timestep-sampling",
                label: "Timestep sampling",
                kind: .choice,
                choices: ["uniform", "bellCurve", "contentFocused", "styleFocused", "logitNormal", "shift"]
            ),
            .init(
                flag: "--timestep-loss-weighting",
                label: "Timestep weighting",
                kind: .choice,
                choices: ["none", "weighted"]
            ),
            .init(flag: "--loss-weighting", label: "Loss weighting", kind: .choice, choices: ["none", "snr", "minSNR"]),
            .init(flag: "--timestep-low", label: "Timestep low", kind: .integer),
            .init(flag: "--timestep-high", label: "Timestep high", kind: .integer),
            .init(flag: "--lr-warmup-steps", label: "LR warmup", kind: .integer),
            .init(flag: "--no-cosine-scheduler", label: "Disable cosine scheduler", kind: .boolean),
            .init(flag: "--lr-min-factor", label: "LR minimum factor", kind: .number),
            .init(flag: "--adam-weight-decay", label: "Adam weight decay", kind: .number),
            .init(flag: "--synthetic-samples", label: "Synthetic samples", kind: .integer),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let imageValidate = MereRunCommandCapability(
        id: "image.validate",
        command: ["image", "validate"],
        title: "Validate image runtime",
        summary: "Run deterministic VAE, encoder, transformer, and pipeline checks.",
        options: [
            .init(
                flag: "--test",
                label: "Suite",
                kind: .choice,
                choices: ["vae", "encoder", "transformer", "pipeline", "all"]
            ),
            .init(flag: "--family", label: "Family", kind: .choice, choices: ["zimage", "klein"]),
            .init(flag: "--output", label: "Output", kind: .directory),
            .init(flag: "--save-reference", label: "Save reference", kind: .boolean),
            .init(flag: "--compare", label: "Compare", kind: .boolean),
            .init(flag: "--reference-dir", label: "Reference directory", kind: .directory)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let imageDatasetDiscover = MereRunCommandCapability(
        id: "image.dataset.discover",
        command: ["image", "dataset", "discover"],
        title: "Discover image datasets",
        summary: "Find trainable image-caption dataset leaves and produce preflight commands.",
        options: [
            .init(flag: "--root", label: "Root", kind: .directory, required: true),
            .init(flag: "--max-depth", label: "Max depth", kind: .integer),
            .init(flag: "--min-usable-pairs", label: "Minimum pairs", kind: .integer),
            .init(flag: "--training-output-root", label: "Training output root", kind: .directory),
            .init(flag: "--training-model", label: "Training model", kind: .string),
            .init(flag: "--training-recipe", label: "Training recipe", kind: .string),
            .init(flag: "--exclude-preview-images", label: "Exclude preview images", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let imageRunPlan = MereRunCommandCapability(
        id: "image.run-plan",
        command: ["image", "run-plan"],
        title: "Run image plan",
        summary: "Preflight, materialize, or execute a saved image workflow plan.",
        arguments: [
            .init(name: "file", label: "Plan", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--materialize", label: "Materialize run", kind: .directory)
        ],
        output: .init(kind: .file)
    )

    public static let imageVisualizeRun = MereRunCommandCapability(
        id: "image.visualize-run",
        command: ["image", "visualize-run"],
        title: "Visualize image run",
        summary: "Open the loopback dashboard for a durable image LoRA run.",
        arguments: [
            .init(name: "run-directory", label: "Run directory", kind: .directory, required: true)
        ],
        options: [
            .init(flag: "--port", label: "Port", kind: .integer)
        ],
        output: .init(kind: .service)
    )

    public static let imageReconstruct3D = MereRunCommandCapability(
        id: "image.reconstruct-3d",
        command: ["image", "reconstruct-3d"],
        title: "TripoSR reconstruction",
        summary: "Reconstruct a colored object mesh from a single image with native TripoSR.",
        arguments: [
            .init(name: "input", label: "Input image", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--resolution", label: "Resolution", kind: .integer),
            .init(flag: "--density-threshold", label: "Density threshold", kind: .number),
            .init(flag: "--foreground-ratio", label: "Foreground ratio", kind: .number),
            .init(flag: "--already-framed", label: "Already framed", kind: .boolean),
            .init(flag: "--no-vertex-colors", label: "Geometry only", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let imageReconstruct3DTrellis2 = MereRunCommandCapability(
        id: "image.reconstruct-3d-trellis2",
        command: ["image", "reconstruct-3d-trellis2"],
        title: "TRELLIS.2 reconstruction",
        summary: "Reconstruct a native 512-resolution PBR O-Voxel asset from one image.",
        arguments: [
            .init(name: "input", label: "Input image", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--texture-seed", label: "Texture seed", kind: .integer),
            .init(flag: "--max-tokens", label: "Maximum sparse tokens", kind: .integer),
            .init(flag: "--already-framed", label: "Already framed", kind: .boolean),
            .init(flag: "--no-remesh", label: "Skip remeshing", kind: .boolean),
            .init(flag: "--remesh-band", label: "Remesh band", kind: .number),
            .init(flag: "--seal-radius", label: "Seal radius", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let imageReconstruct3DMultiview = MereRunCommandCapability(
        id: "image.reconstruct-3d-multiview",
        command: ["image", "reconstruct-3d-multiview"],
        title: "InstantMesh multiview reconstruction",
        summary: "Reconstruct a colored mesh from four or six ordered source views.",
        options: [
            .init(flag: "--view", label: "View", kind: .file, repeatable: true),
            .init(flag: "--output", label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--cameras", label: "Cameras", kind: .file),
            .init(flag: "--resolution", label: "Resolution", kind: .integer),
            .init(flag: "--no-vertex-colors", label: "Geometry only", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )
}
