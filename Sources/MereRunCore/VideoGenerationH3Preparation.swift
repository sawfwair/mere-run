import Foundation

/// Resolves H3 adapter identity and compatibility without loading the model.
public struct VideoGenerationH3Preparation: Sendable {
    public let adapterURL: URL?

    public init(
        options: VideoGenerationOptions,
        profile: VideoGenerationModelProfile,
        modelRoot: URL?,
        adaptersRoot: URL = MereRunModelPaths.adaptersDir,
        fileManager: FileManager = .default,
        requireInstalled: Bool = true
    ) throws {
        let baseModelID: String
        if profile == .h3Ref2VA {
            baseModelID = ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue
        } else if options.resolvedRequestedModel == ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue {
            baseModelID = ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue
        } else {
            baseModelID = ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
        }
        let reference = options.h3Adapter ?? (options.usesEmbeddedFastH3Adapter
            ? modelRoot.map { MiniMaxH3Resources(rootURL: $0).fastH3AdapterURL.path }
                ?? MiniMaxH3TurboAdapter.fastH3VSADataFreeFilename
            : nil)
        adapterURL = try ManagedAdapterArgumentResolver.resolve(
            reference, baseModelID: baseModelID, adaptersRoot: adaptersRoot,
            fileManager: fileManager, requireInstalled: requireInstalled
        ).map { URL(fileURLWithPath: $0).standardizedFileURL }
        guard let adapterURL else { return }
        let recipe = MiniMaxH3TurboAdapter.inferenceRecipe(for: adapterURL)
        let task = profile == .h3Ref2VA ? "ref2va" : "fl2va"
        guard recipe.supports(task: task) else {
            throw Self.issue("h3_adapter_task_mismatch", "MiniMax-H3 adapter \(recipe.name) requires \(recipe.task.rawValue), not \(task).")
        }
        if recipe.task == .fl2va {
            let supportsTurbo: Bool
            if let modelRoot {
                supportsTurbo = try MiniMaxH3Resources(rootURL: modelRoot).transformerStorage().supportsFL2VATurboAdapters
            } else {
                supportsTurbo = options.resolvedRequestedModel == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
                    || options.resolvedRequestedModel == MiniMaxH3Resources.fl2vaQ8ModelID
                    || options.resolvedRequestedModel == MiniMaxH3Resources.fastH3ModelID
            }
            guard supportsTurbo else {
                throw Self.issue("h3_adapter_requires_bf16_or_q8", "MiniMax-H3 FL2VA adapters require compact BF16 or Q8; legacy Q4 is unsupported.")
            }
        }
        if recipe.task == .ref2va, options.h3WeightMode == "quantized" {
            throw Self.issue("h3_ref2va_adapter_requires_resident_bf16", "MiniMax-H3 Ref2VA Turbo requires resident BF16 weights; use --h3-weight-mode resident-bf16.")
        }
        if let steps = options.steps, !recipe.supports(schedulePointCount: steps) {
            let supported = recipe.supportedSchedulePointCounts.sorted().map(String.init).joined(separator: " or ")
            throw Self.issue("h3_adapter_steps_invalid", "Omit --steps or set --steps \(supported) schedule points for \(recipe.name).")
        }
    }

    private static func issue(_ id: String, _ message: String) -> VideoGenerationIssue {
        VideoGenerationIssue(id: id, title: "MiniMax-H3 adapter is incompatible", message: message)
    }
}
