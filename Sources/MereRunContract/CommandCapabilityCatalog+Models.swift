import Foundation

extension MereRunCapabilityCatalog {
    public static let modelStorage = MereRunCommandCapability(
        id: "model.storage",
        command: ["model", "storage"],
        title: "Model storage",
        summary: "Inspect physical storage, sharing, and reclaimable bytes.",
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelGarbageCollect = MereRunCommandCapability(
        id: "model.gc",
        command: ["model", "gc"],
        title: "Model storage cleanup",
        summary: "Plan or execute safe cleanup of unreferenced payloads and partial downloads.",
        options: [
            .init(flag: "--force", label: "Execute", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelRuntimeGet = MereRunCommandCapability(
        id: "model.runtime.get",
        command: ["model", "runtime", "get"],
        title: "Read runtime policy",
        summary: "Read typed API residency and default generation settings.",
        arguments: [
            .init(name: "model", label: "Model or alias", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelRuntimeSet = MereRunCommandCapability(
        id: "model.runtime.set",
        command: ["model", "runtime", "set"],
        title: "Set runtime policy",
        summary: "Update typed API residency and default generation settings.",
        arguments: [
            .init(name: "model", label: "Model or alias", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--alias", label: "Alias", kind: .string),
            .init(flag: "--clear-alias", label: "Clear alias", kind: .boolean),
            .init(flag: "--pinned", label: "Pin model", kind: .boolean),
            .init(flag: "--unpinned", label: "Unpin model", kind: .boolean),
            .init(flag: "--ttl-seconds", label: "TTL", kind: .integer),
            .init(flag: "--clear-ttl", label: "Clear TTL", kind: .boolean),
            .init(flag: "--max-context-tokens", label: "Max context", kind: .integer),
            .init(flag: "--clear-max-context-tokens", label: "Clear max context", kind: .boolean),
            .init(flag: "--max-tokens", label: "Max output", kind: .integer),
            .init(flag: "--clear-max-tokens", label: "Clear max output", kind: .boolean),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--clear-temperature", label: "Clear temperature", kind: .boolean),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--clear-top-p", label: "Clear top-p", kind: .boolean),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--clear-min-p", label: "Clear min-p", kind: .boolean),
            .init(
                flag: "--engine",
                label: "Engine",
                kind: .choice,
                choices: [
                    "text-code",
                    "text-chat-klein",
                    "text-chat-gemma4",
                    "text-chat-diffusiongemma",
                    "text-chat-laguna",
                    "text-chat-q36",
                    "text-chat-q35",
                    "text-chat-lfm2",
                    "text-chat-deepseek-v4-flash",
                    "text-chat-muse-glimmer",
                    "text-chat-nemotron-h",
                    "text-chat-nemotron-omni"
                ]
            ),
            .init(flag: "--clear-engine", label: "Clear engine", kind: .boolean),
            .init(
                flag: "--kv-cache-mode",
                label: "KV cache",
                kind: .choice,
                choices: ["default", "affine4", "affine8", "polar2", "auto"]
            ),
            .init(flag: "--clear-kv-cache-mode", label: "Clear KV cache", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelList = MereRunCommandCapability(
        id: "model.list",
        command: ["model", "list"],
        title: "List models",
        summary: "List managed model install state.",
        options: [
            .init(flag: "--measure-sizes", label: "Measure referenced sizes", kind: .boolean),
            .init(flag: "--json", label: "JSON inventory", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelCapabilities = MereRunCommandCapability(
        id: "model.capabilities",
        command: ["model", "capabilities"],
        title: "Model capabilities",
        summary: "Inspect hardware support and setup recommendations.",
        options: [
            .init(flag: "--all", label: "Include unsupported", kind: .boolean),
            .init(flag: "--recommended", label: "Recommended only", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelPull = MereRunCommandCapability(
        id: "model.pull",
        command: ["model", "pull"],
        title: "Pull model",
        summary: "Preflight or install one or all managed models.",
        arguments: [
            .init(name: "target", label: "Model", kind: .string, required: false)
        ],
        options: [
            .init(
                flag: "--cache-dir", label: "Download cache", kind: .directory, group: Group.output, tier: .expert
            ),
            .init(flag: "--all", label: "All models", kind: .boolean),
            .init(flag: "--force", label: "Force download", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean),
            .init(flag: "--allow-unsupported", label: "Allow unsupported", kind: .boolean),
            .init(flag: "--accept-model-license", label: "Accept model terms", kind: .boolean),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelInfo = MereRunCommandCapability(
        id: "model.info",
        command: ["model", "info"],
        title: "Model info",
        summary: "Inspect a model manifest and resolved components.",
        arguments: [
            .init(name: "target", label: "Model", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--components", label: "Components", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelRemove = MereRunCommandCapability(
        id: "model.remove",
        command: ["model", "remove"],
        title: "Remove model",
        summary: "Remove a managed model with optional cache preservation and receipt.",
        arguments: [
            .init(name: "target", label: "Model", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--force", label: "Force", kind: .boolean),
            .init(flag: "--keep-cache", label: "Keep cache", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelRepairManifests = MereRunCommandCapability(
        id: "model.repair-manifests",
        command: ["model", "repair-manifests"],
        title: "Repair manifests",
        summary: "Restore missing manifests for known local models.",
        options: [
            .init(
                flag: "--accept-model-license", label: "Accept model terms", kind: .boolean, group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .expert
            ),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelOptimize = MereRunCommandCapability(
        id: "model.optimize",
        command: ["model", "optimize"],
        title: "Optimize model",
        summary: "Build inference-only caches for a supported installed model.",
        arguments: [
            .init(name: "target", label: "Model or local root", kind: .string, required: true)
        ],
        options: [
            .init(
                flag: "--text-encoder-only", label: "Optimize text encoder only", kind: .boolean, group: Group.run,
                tier: .expert
            ),
            .init(flag: "--force", label: "Replace cache", kind: .boolean),
            .init(flag: "--output", label: "Standalone checkpoint", kind: .directory),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text, flag: "--output", optional: true)
    )

    public static let modelLocationList = MereRunCommandCapability(
        id: "model.location.list",
        command: ["model", "location", "list"],
        title: "List model locations",
        summary: "List the writable store, read-only search roots, and explicit bindings.",
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelLocationAdd = MereRunCommandCapability(
        id: "model.location.add",
        command: ["model", "location", "add"],
        title: "Add search root",
        summary: "Register a read-only root containing directories named for canonical model ids.",
        arguments: [
            .init(name: "path", label: "Search root", kind: .directory, required: true)
        ],
        options: [],
        output: .init(kind: .text)
    )

    public static let modelLocationRemove = MereRunCommandCapability(
        id: "model.location.remove",
        command: ["model", "location", "remove"],
        title: "Remove search root",
        summary: "Unregister a read-only search root without deleting its files.",
        arguments: [
            .init(name: "path", label: "Search root", kind: .directory, required: true)
        ],
        options: [],
        output: .init(kind: .text)
    )

    public static let modelLocationBind = MereRunCommandCapability(
        id: "model.location.bind",
        command: ["model", "location", "bind"],
        title: "Bind model directory",
        summary: "Bind a canonical model id to an arbitrary read-only directory.",
        arguments: [
            .init(name: "modelID", label: "Model id", kind: .string, required: true),
            .init(name: "path", label: "Model directory", kind: .directory, required: true)
        ],
        options: [
            .init(
                flag: "--accept-model-license",
                label: "Accept model license",
                kind: .boolean
            )
        ],
        output: .init(kind: .text)
    )

    public static let modelLocationUnbind = MereRunCommandCapability(
        id: "model.location.unbind",
        command: ["model", "location", "unbind"],
        title: "Unbind model directory",
        summary: "Remove explicit bindings without deleting model files.",
        arguments: [
            .init(name: "modelID", label: "Model id", kind: .string, required: true),
            .init(name: "path", label: "Model directory", kind: .directory, required: false)
        ],
        options: [],
        output: .init(kind: .text)
    )

    // MARK: - Model benchmarks
}
