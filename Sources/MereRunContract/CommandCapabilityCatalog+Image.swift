import Foundation

extension MereRunCapabilityCatalog {
    private typealias ImageF = ImageGenerateFamily

    /// Families whose runtime takes an input image (`--input`, and with it `--mask` and
    /// `--outpaint`); FLUX.1, Krea 2, and Ideogram 4 are text-to-image only.
    private static var imageInputScope: MereRunOptionScope<ImageF> { ImageF.except(.flux1, .krea, .ideogram) }
    /// Families that load `--lora` adapters.
    private static let imageLoRAFamilies: [ImageF] = [.flux1, .klein, .flux2Dev, .zimage, .krea]

    public static let imageGenerate = MereRunCommandCapability(
        id: "image.generate",
        command: ["image", "generate"],
        title: "Generate and edit images",
        summary: "Generate, transform, or personalize images with references, structured prompts, and LoRAs.",
        options: [
            .init(flag: "--run-dir", label: "Run directory", kind: .directory, group: Group.output, tier: .expert),
            .init(
                flag: "--sigmas", label: "Sigma schedule", kind: .string, group: Group.sampling, tier: .expert
            ).scoped(ImageF.only(.klein, .flux2Dev)),
            .init(flag: "--prompt", aliases: ["-p"], label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(flag: "--negative-prompt", aliases: ["-n"], label: "Negative prompt", kind: .string, group: Group.prompt, tier: .standard)
                // FLUX.1 refuses a negative prompt with text in it; an empty one reads as omitted.
                // Qwen-Image-Edit Lightning runs without guidance, so it never reads one.
                .scoped(ImageF.except(.flux1, ignoredBy: [.flux2Dev, .krea, .qwenEditLightning, .ideogram])),
            .init(
                flag: "--cfg", aliases: ["--cfg-scale"], label: "CFG scale", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 20, step: 0.5)
            ).scoped(ImageF.except(ignoredBy: [.krea]), .rule(.qwenEditLightning, range: .init(min: 1, max: 1))),
            .init(
                flag: "--sigma-shift", label: "Sigma shift", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 16, step: 0.1)
            ).scoped(ImageF.except(ignoredBy: [.flux1, .hidream, .qwenEdit, .qwenEditLightning, .ideogram])),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--width", aliases: ["-W"], label: "Width", kind: .integer,
                defaultValue: "1024", group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--height", aliases: ["-H"], label: "Height", kind: .integer,
                defaultValue: "1024", group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--steps", aliases: ["-s"], label: "Steps", kind: .integer,
                group: Group.sampling, tier: .essential, range: .init(min: 1, max: 100, step: 1)
            ).scoped(ImageF.rule(.qwen21, range: .init(min: 2, step: 1)), .rule(.qwenEditLightning, values: ["4"])),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--input", aliases: ["-i"], label: "Input image", kind: .file, group: Group.inputs, tier: .standard)
                .scoped(imageInputScope),
            .init(flag: "--mask", label: "Edit mask", kind: .file, group: Group.inputs, tier: .standard, dependsOn: "--input")
                .scoped(imageInputScope),
            .init(
                flag: "--outpaint", label: "Outpaint padding", kind: .string,
                group: Group.inputs, tier: .expert, dependsOn: "--input"
            ).scoped(imageInputScope),
            .init(
                flag: "--mask-feather", label: "Mask feather", kind: .integer,
                defaultValue: "8", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 128, step: 1), dependsOn: "--input"
            ).scoped(ImageF.except(ignoredBy: [.flux1, .krea, .ideogram])),
            .init(
                flag: "--ref-image", label: "Reference image", kind: .file, repeatable: true,
                group: Group.inputs, tier: .standard
            ).scoped(
                ImageF.except(.flux1, .zimage, .krea, .ideogram),
                // Klein keeps the first four references, counting --input, and drops the rest.
                .rule(.klein, maxCount: 4, severity: .warning),
                .rule(.flux2Dev, maxCount: 4, severity: .warning),
                // Qwen-Image-Edit's limit of three counts distinct files, so Core checks it.
                .rule(.qwen21, maxCount: 10)
            ),
            .init(
                flag: "--keep-original-aspect", label: "Keep original aspect", kind: .boolean,
                group: Group.inputs, tier: .expert, dependsOn: "--ref-image"
            ).scoped(ImageF.only(.hidream, ignoredBy: ImageF.allCases.filter { $0 != .hidream })),
            .init(
                flag: "--strength", aliases: ["--str"], label: "Edit strength", kind: .number,
                group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05)
            ).scoped(ImageF.except(
                .qwen21,
                ignoredBy: [.flux1, .hidream, .sensenova, .krea, .qwenEdit, .qwenEditLightning, .ideogram]
            )),
            .init(
                flag: "--max-sequence-length", label: "Max sequence length", kind: .integer,
                defaultValue: "512", group: Group.sampling, tier: .expert, range: .init(min: 64, max: 4_096, step: 64)
            ).scoped(
                ImageF.only(
                    .flux1, .zimage, .krea, .ideogram,
                    ignoredBy: [.klein, .flux2Dev, .hidream, .sensenova, .qwen21, .qwenEdit, .qwenEditLightning]
                ),
                // Longer values are clamped to what the text encoder reads.
                .rule(.flux1, range: .init(max: 512), severity: .warning),
                .rule(.krea, range: .init(max: 512), severity: .warning),
                .rule(.ideogram, range: .init(max: 2_048), severity: .warning)
            ),
            .init(flag: "--structured-prompt", aliases: ["--json-prompt"], label: "Structured prompt", kind: .boolean, group: Group.prompt, tier: .standard),
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
            .init(flag: "--lora", aliases: ["-l"], label: "LoRA", kind: .file, repeatable: true, group: Group.modelAndAdapters, tier: .standard)
                .scoped(
                    ImageF.only(.flux1, .klein, .flux2Dev, .zimage, .krea),
                    .rule(.zimage, maxCount: 1),
                    .rule(.krea, maxCount: 1)
                ),
            .init(
                flag: "--lora-scale", label: "LoRA scale", kind: .number,
                defaultValue: "1.0", group: Group.modelAndAdapters, tier: .standard,
                range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--lora"
            ).scoped(ImageF.except(ignoredBy: ImageF.allCases.filter { !imageLoRAFamilies.contains($0) })),
            .init(
                flag: "--krea-conditioning-multiplier", label: "Krea conditioning", kind: .number,
                group: Group.sampling, tier: .expert
            ).scoped(ImageF.only(.krea, ignoredBy: ImageF.allCases.filter { $0 != .krea })),
            .init(
                flag: "--krea-conditioning-layer-weights", label: "Krea layer weights", kind: .string,
                group: Group.sampling, tier: .expert
            ).scoped(ImageF.only(.krea, ignoredBy: ImageF.allCases.filter { $0 != .krea })),
            .init(
                flag: "--krea-base-quantization-bits",
                label: "Krea base quantization",
                kind: .choice,
                choices: ["4", "8"],
                group: Group.modelAndAdapters, tier: .expert
            ).scoped(ImageF.only(.krea)),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "png", flag: "--output"),
        routing: imageGenerateRouting
    )

    private typealias TrainF = ImageTrainLoRAFamily

    /// Options only the FLUX.2 Klein trainer reads; the Krea 2 trainer refuses each of them.
    private static var kleinTrainingOnly: MereRunOptionScope<TrainF> { TrainF.only(.klein) }
    /// Options only the Krea 2 trainer reads; the Klein trainer refuses each of them.
    private static var kreaTrainingOnly: MereRunOptionScope<TrainF> { TrainF.only(.krea) }
    /// Klein options the CLI checks on Krea by value: Krea runs without the default (warns) and
    /// refuses any other value. Each carries a Krea rule naming that default.
    private static var kleinOnlyUnlessDefault: MereRunOptionScope<TrainF> { TrainF.only(.klein, ignoredBy: [.krea]) }
    /// `--recipe` is trimmed and lowercased, and older recipe names still resolve
    /// (`ImageLoRATrainingOptions.resolveLoRATrainingRecipe`).
    private static let trainingRecipeSpellings = MereRunChoiceSpellings(ignoresCase: true, aliases: [
        "local-krea-style": "krea-fast-style",
        "fal-krea-style": "krea-fast-style",
        "krea2-fast-style": "krea-fast-style",
        "krea-movie-style": "krea-cinematic-style",
        "krea-wide-style": "krea-cinematic-style",
        "local-klein-style": "klein-fast-style",
        "flux2-klein-fast-style": "klein-fast-style"
    ])

    public static let imageTrainLoRA = MereRunCommandCapability(
        id: "image.train-lora",
        command: ["image", "train-lora"],
        title: "Train image LoRA",
        summary: "Train Krea 2 or FLUX.2 Klein adapters with recipes, previews, checkpoints, and dashboards.",
        options: [
            // Krea 2 also runs without a dataset when --synthetic-samples is set; Core checks that.
            .init(flag: "--data", aliases: ["-d"], label: "Dataset", kind: .directory)
                .scoped(TrainF.rule(.klein, required: true)),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, required: true),
            .init(flag: "--model", aliases: ["-m"], label: "Base model", kind: .string),
            .init(flag: "--width", aliases: ["-W"], label: "Width", kind: .integer),
            .init(flag: "--height", aliases: ["-H"], label: "Height", kind: .integer),
            .init(flag: "--training-steps", aliases: ["--steps"], label: "Training steps", kind: .integer),
            .init(flag: "--batch-size", label: "Batch size", kind: .integer),
            .init(flag: "--learning-rate", aliases: ["--lr"], label: "Learning rate", kind: .number),
            .init(flag: "--rank", label: "Rank", kind: .integer),
            .init(flag: "--alpha", label: "Alpha", kind: .number),
            .init(flag: "--max-text-length", label: "Max text length", kind: .integer),
            .init(flag: "--scheduler-steps", label: "Scheduler steps", kind: .integer),
            .init(flag: "--caption-dropout", label: "Caption dropout", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--lite", label: "Lite targets", kind: .boolean),
            .init(flag: "--base-quantization-bits", label: "Base quantization", kind: .choice, choices: ["4", "8"])
                .scoped(kreaTrainingOnly),
            .init(flag: "--exclude-preview-images", label: "Exclude preview images", kind: .boolean),
            .init(flag: "--checkpoint-interval", label: "Checkpoint interval", kind: .integer).scoped(kleinTrainingOnly),
            .init(flag: "--resume-from", label: "Resume checkpoint", kind: .file).scoped(kleinTrainingOnly),
            .init(flag: "--max-resolution", label: "Max resolution", kind: .integer).scoped(kleinTrainingOnly),
            .init(flag: "--progressive", label: "Progressive resolution", kind: .boolean).scoped(kleinTrainingOnly),
            .init(flag: "--low-ram", label: "Low RAM", kind: .boolean).scoped(kleinTrainingOnly),
            .init(flag: "--no-compile", label: "Disable compile", kind: .boolean),
            .init(flag: "--gradient-checkpointing", label: "Gradient checkpointing", kind: .boolean).scoped(kleinTrainingOnly),
            .init(
                flag: "--recipe",
                label: "Recipe",
                kind: .choice,
                choices: ["krea-fast-style", "krea-cinematic-style", "klein-fast-style"],
                choiceSpellings: trainingRecipeSpellings
            ).scoped(TrainF.rule(.krea, values: ["krea-fast-style", "krea-cinematic-style"])),
            .init(flag: "--benchmark-steps", label: "Benchmark steps", kind: .integer).scoped(kleinTrainingOnly),
            .init(flag: "--benchmark-warmup-steps", label: "Benchmark warmup", kind: .integer)
                .scoped(kleinOnlyUnlessDefault, .rule(.krea, range: .init(min: 5, max: 5))),
            .init(flag: "--sample-interval", label: "Sample interval", kind: .integer).scoped(kleinTrainingOnly),
            .init(flag: "--sample-prompt", label: "Sample prompt", kind: .string).scoped(kleinTrainingOnly),
            .init(flag: "--sample-model", label: "Sample model", kind: .string).scoped(kleinTrainingOnly),
            .init(flag: "--sample-steps", label: "Sample steps", kind: .integer)
                .scoped(kleinOnlyUnlessDefault, .rule(.krea, range: .init(min: 8, max: 8))),
            .init(flag: "--sample-cfg", label: "Sample CFG", kind: .number)
                .scoped(kleinOnlyUnlessDefault, .rule(.krea, range: .init(min: 1, max: 1))),
            .init(flag: "--sample-lora-scale", label: "Sample LoRA scale", kind: .number)
                .scoped(kleinOnlyUnlessDefault, .rule(.krea, range: .init(min: 1, max: 1))),
            .init(flag: "--sample-seed", label: "Sample seed", kind: .integer).scoped(kleinTrainingOnly),
            .init(flag: "--visualize", label: "Visualize", kind: .boolean),
            .init(flag: "--visualize-port", label: "Visualization port", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--lora-target-ranks", label: "Target ranks", kind: .string).scoped(kleinTrainingOnly),
            .init(flag: "--lora-rank-preset", label: "Rank preset", kind: .choice, choices: ["flux2-style-128"])
                .scoped(kleinTrainingOnly),
            .init(flag: "--lora-target-preset", label: "Target preset", kind: .choice, choices: ["fal-klein-fast"])
                .scoped(kleinTrainingOnly),
            .init(
                flag: "--lora-target-mode",
                label: "Target mode",
                kind: .choice,
                choices: ["suffix", "transformer-linear-walk"]
            ).scoped(kleinTrainingOnly),
            .init(
                flag: "--timestep-sampling",
                label: "Timestep sampling",
                kind: .choice,
                choices: ["uniform", "bellCurve", "contentFocused", "styleFocused", "logitNormal", "shift"]
            ).scoped(kleinTrainingOnly),
            .init(
                flag: "--timestep-loss-weighting",
                label: "Timestep weighting",
                kind: .choice,
                choices: ["none", "weighted"]
            ).scoped(kleinTrainingOnly),
            .init(flag: "--loss-weighting", label: "Loss weighting", kind: .choice, choices: ["none", "snr", "minSNR"])
                .scoped(kleinTrainingOnly),
            .init(flag: "--timestep-low", label: "Timestep low", kind: .integer).scoped(kleinTrainingOnly),
            .init(flag: "--timestep-high", label: "Timestep high", kind: .integer).scoped(kleinTrainingOnly),
            .init(flag: "--lr-warmup-steps", label: "LR warmup", kind: .integer),
            .init(flag: "--no-cosine-scheduler", label: "Disable cosine scheduler", kind: .boolean),
            .init(flag: "--lr-min-factor", label: "LR minimum factor", kind: .number),
            .init(flag: "--adam-weight-decay", label: "Adam weight decay", kind: .number).scoped(kleinTrainingOnly),
            .init(flag: "--synthetic-samples", label: "Synthetic samples", kind: .integer).scoped(kreaTrainingOnly),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output"),
        routing: imageTrainLoRARouting
    )

    public static let imageValidate = MereRunCommandCapability(
        id: "image.validate",
        command: ["image", "validate"],
        title: "Validate image runtime",
        summary: "Run deterministic VAE, encoder, transformer, and pipeline checks.",
        options: [
            .init(
                flag: "--test", aliases: ["-t"],
                label: "Suite",
                kind: .choice,
                choices: ["vae", "encoder", "transformer", "pipeline", "all"]
            ),
            .init(flag: "--family", aliases: ["-m"], label: "Family", kind: .choice, choices: ["zimage", "klein"]),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .directory),
            .init(flag: "--save-reference", label: "Save reference", kind: .boolean),
            .init(flag: "--compare", label: "Compare", kind: .boolean),
            .init(flag: "--reference-dir", label: "Reference directory", kind: .directory)
        ],
        output: .init(kind: .directory, flag: "--output"),
        routing: imageValidateRouting
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
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--resolution", label: "Resolution", kind: .integer, group: Group.run, tier: .essential),
            .init(flag: "--density-threshold", label: "Density threshold", kind: .number),
            .init(flag: "--foreground-ratio", label: "Foreground ratio", kind: .number),
            .init(flag: "--already-framed", label: "Already framed", kind: .boolean),
            .init(flag: "--no-vertex-colors", label: "Geometry only", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output"),
        routing: imageReconstruct3DRouting
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
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential),
            .init(flag: "--texture-seed", label: "Texture seed", kind: .integer),
            .init(flag: "--max-tokens", label: "Maximum sparse tokens", kind: .integer),
            .init(flag: "--already-framed", label: "Already framed", kind: .boolean),
            .init(flag: "--no-remesh", label: "Skip remeshing", kind: .boolean),
            .init(flag: "--remesh-band", label: "Remesh band", kind: .number),
            .init(flag: "--seal-radius", label: "Seal radius", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output"),
        routing: imageReconstruct3DTrellis2Routing
    )

    public static let imageReconstruct3DMultiview = MereRunCommandCapability(
        id: "image.reconstruct-3d-multiview",
        command: ["image", "reconstruct-3d-multiview"],
        title: "InstantMesh multiview reconstruction",
        summary: "Reconstruct a colored mesh from four or six ordered source views.",
        options: [
            .init(flag: "--view", label: "View", kind: .file, repeatable: true),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--cameras", label: "Cameras", kind: .file),
            .init(flag: "--resolution", label: "Resolution", kind: .integer, group: Group.run, tier: .essential),
            .init(flag: "--no-vertex-colors", label: "Geometry only", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output"),
        routing: imageReconstruct3DMultiviewRouting
    )
}
