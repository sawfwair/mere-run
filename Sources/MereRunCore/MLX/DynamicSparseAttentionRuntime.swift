import Foundation
import MLX

enum DynamicSparseAttentionModel: String, CaseIterable, Sendable {
    case wan2
    case scail2
    case ltx
    case cosmos3
    case trellis2
    case flux2
    case krea2
    case qwenImageEdit = "qwen-image-edit"
    case zImage = "z-image"
    case ideogram4
}

/// Opt-in staging controller for model-specific dynamic sparse attention.
///
/// H3 owns its released admission policy. Other model integrations stay
/// disabled unless explicitly selected through `MERERUN_DYNAMIC_SPARSE_ATTENTION`.
/// This lets source and CPU-only validation land independently from the real
/// Metal numerical and artifact gates required before production enablement.
final class DynamicSparseAttentionRuntime {
    let model: DynamicSparseAttentionModel
    let policy: DynamicSparseAttentionPolicy
    let maximumQueryTokens: Int
    let maximumKernelsPerEvaluation: Int

    var logHandler: ((String) -> Void)?

    private(set) var stepIndex = 0
    private(set) var stepCount = 0
    private var gateResults: [String: Bool] = [:]

    init(
        model: DynamicSparseAttentionModel,
        policy: DynamicSparseAttentionPolicy,
        maximumQueryTokens: Int = 1_024,
        maximumKernelsPerEvaluation: Int = 4
    ) {
        precondition(maximumQueryTokens > 0)
        precondition(maximumKernelsPerEvaluation > 0)
        self.model = model
        self.policy = policy
        self.maximumQueryTokens = maximumQueryTokens
        self.maximumKernelsPerEvaluation = maximumKernelsPerEvaluation
    }

    static func configured(
        model: DynamicSparseAttentionModel,
        minimumSequenceLength: Int = 12_000,
        denseLeadingStepFraction: Float = 0.2,
        denseTrailingStepCount: Int = 1,
        denseLeadingLayerCount: Int = 2,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> DynamicSparseAttentionRuntime? {
        guard selection(environment: environment).contains(model) else { return nil }
        let threshold = Float(environment["MERERUN_DYNAMIC_SPARSE_TAU"] ?? "")
            .flatMap { $0 >= 0 ? $0 : nil } ?? 1
        let sequenceThreshold = Int(environment["MERERUN_DYNAMIC_SPARSE_MIN_TOKENS"] ?? "")
            .flatMap { $0 > 0 ? $0 : nil } ?? minimumSequenceLength
        return DynamicSparseAttentionRuntime(
            model: model,
            policy: DynamicSparseAttentionPolicy(
                thresholdStandardDeviations: threshold,
                minimumSequenceLength: sequenceThreshold,
                denseLeadingStepFraction: denseLeadingStepFraction,
                denseTrailingStepCount: denseTrailingStepCount,
                denseLeadingLayerCount: denseLeadingLayerCount
            )
        )
    }

    static func selection(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Set<DynamicSparseAttentionModel> {
        guard let raw = environment["MERERUN_DYNAMIC_SPARSE_ATTENTION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
              !raw.isEmpty,
              raw != "0",
              raw != "false" else { return [] }
        if ["1", "true", "all"].contains(raw) {
            return Set(DynamicSparseAttentionModel.allCases)
        }
        return Set(raw.split(separator: ",").compactMap { token in
            let value = token.trimmingCharacters(in: .whitespacesAndNewlines)
            return DynamicSparseAttentionModel.allCases.first {
                $0.rawValue == value
            }
        })
    }

    func beginStep(index: Int, count: Int) {
        precondition(count > 0)
        precondition((0..<count).contains(index))
        stepIndex = index
        stepCount = count
    }

    func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        layerIndex: Int,
        prefixTokenCount: Int,
        scale: Float
    ) -> MLXArray? {
        guard let request = policy.request(
            stepIndex: stepIndex,
            stepCount: stepCount,
            layerIndex: layerIndex,
            sequenceLength: queries.dim(2),
            prefixTokenCount: prefixTokenCount
        ) else { return nil }

        let key = gateKey(
            queries: queries,
            keys: keys,
            values: values,
            prefixTokenCount: prefixTokenCount
        )
        if gateResults[key] == nil {
            let gate = DynamicSparseAttention.denseRouteGate(
                queries: queries,
                keys: keys,
                values: values,
                queryStart: prefixTokenCount,
                scale: scale
            )
            gateResults[key] = gate?.passed ?? false
            if let gate {
                logHandler?(String(
                    format: "dynamic_sparse model=%@ gate=%@ max_abs=%.6g mean_abs=%.6g rel_l2=%.6g",
                    model.rawValue,
                    gate.passed ? "pass" : "fail",
                    gate.maximumAbsoluteError,
                    gate.meanAbsoluteError,
                    gate.relativeL2Error
                ))
            } else {
                logHandler?("dynamic_sparse model=\(model.rawValue) gate=unavailable key=\(key)")
            }
        }
        guard gateResults[key] == true else { return nil }
        return DynamicSparseAttention.call(
            queries: queries,
            keys: keys,
            values: values,
            request: request,
            scale: scale,
            maximumQueryTokens: maximumQueryTokens,
            maximumKernelsPerEvaluation: maximumKernelsPerEvaluation
        )
    }

    private func gateKey(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        prefixTokenCount: Int
    ) -> String {
        [
            model.rawValue,
            queries.shape.description,
            String(describing: queries.dtype),
            String(describing: keys.dtype),
            String(describing: values.dtype),
            String(prefixTokenCount),
        ].joined(separator: ":")
    }
}
