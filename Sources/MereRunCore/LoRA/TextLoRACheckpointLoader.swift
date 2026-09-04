import Foundation
import MLX

public struct TextLoRACheckpointLoadReport: Sendable, Hashable {
    public let completedSteps: Int
    public let layerCount: Int
    public let usedLegacyStepOverride: Bool

    public init(completedSteps: Int, layerCount: Int, usedLegacyStepOverride: Bool) {
        self.completedSteps = completedSteps
        self.layerCount = layerCount
        self.usedLegacyStepOverride = usedLegacyStepOverride
    }
}

public enum TextLoRACheckpointLoader {
    public static let checkpointSchema = "mererun.text-lora-checkpoint.v1"

    public static func load(
        from checkpointURL: URL,
        into loraLayers: [String: TrainableLoRALayer],
        config: TextLoRATrainingConfig,
        expectedMetadata: [String: String]
    ) throws -> TextLoRACheckpointLoadReport {
        guard FileManager.default.fileExists(atPath: checkpointURL.path) else {
            throw TextLoRACheckpointError.fileNotFound(checkpointURL.path)
        }

        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: checkpointURL)
        guard metadata["has_optimizer_state"] == "true" else {
            throw TextLoRACheckpointError.optimizerStateRequired
        }

        try validateMetadata(metadata, expected: expectedMetadata)
        try validateRankAndAlpha(metadata, loraLayers: loraLayers)

        let expectedFingerprint = configFingerprint(
            config: config,
            loraLayers: loraLayers,
            metadata: expectedMetadata
        )
        let sidecar = try LoRATrainingCheckpointState.load(nextTo: checkpointURL)
        let completedSteps = try resolveCompletedSteps(
            metadata: metadata,
            sidecar: sidecar,
            override: config.resumeStep
        )
        try validateResumeState(
            sidecar,
            metadata: metadata,
            expectedMetadata: expectedMetadata,
            expectedFingerprint: expectedFingerprint,
            config: config,
            completedSteps: completedSteps
        )

        let expectedKeys = Set(loraLayers.keys.flatMap(Self.tensorKeys(for:)))
        let actualKeys = Set(arrays.keys)
        let missingKeys = expectedKeys.subtracting(actualKeys).sorted()
        let unexpectedKeys = actualKeys.subtracting(expectedKeys).sorted()
        guard missingKeys.isEmpty, unexpectedKeys.isEmpty else {
            throw TextLoRACheckpointError.tensorInventoryMismatch(
                missing: missingKeys,
                unexpected: unexpectedKeys
            )
        }

        var loadedArrays: [MLXArray] = []
        loadedArrays.reserveCapacity(expectedKeys.count)
        for (path, layer) in loraLayers.sorted(by: { $0.key < $1.key }) {
            let keys = tensorKeys(for: path)
            let down = try requiredArray(arrays, key: keys[0], shape: layer.loraDown.shape)
            let up = try requiredArray(arrays, key: keys[1], shape: layer.loraUp.shape)
            let downM = try requiredOptimizerArray(
                arrays,
                key: keys[2],
                shape: layer.loraDown.shape
            )
            let downV = try requiredOptimizerArray(
                arrays,
                key: keys[3],
                shape: layer.loraDown.shape
            )
            let upM = try requiredOptimizerArray(
                arrays,
                key: keys[4],
                shape: layer.loraUp.shape
            )
            let upV = try requiredOptimizerArray(
                arrays,
                key: keys[5],
                shape: layer.loraUp.shape
            )

            layer.loraDown = down.asType(.float32)
            layer.loraUp = up.asType(.float32)
            layer.loraDownM = downM.asType(.float32)
            layer.loraDownV = downV.asType(.float32)
            layer.loraUpM = upM.asType(.float32)
            layer.loraUpV = upV.asType(.float32)
            loadedArrays.append(contentsOf: [
                layer.loraDown,
                layer.loraUp,
                layer.loraDownM!,
                layer.loraDownV!,
                layer.loraUpM!,
                layer.loraUpV!,
            ])
        }
        eval(loadedArrays)

        return TextLoRACheckpointLoadReport(
            completedSteps: completedSteps,
            layerCount: loraLayers.count,
            usedLegacyStepOverride: sidecar == nil && metadata["completed_steps"] == nil
        )
    }

    public static func configFingerprint(
        config: TextLoRATrainingConfig,
        loraLayers: [String: TrainableLoRALayer],
        metadata: [String: String]
    ) -> String {
        let firstLayer = loraLayers.sorted(by: { $0.key < $1.key }).first?.value
        let fields = [
            "format=\(metadata["format"] ?? "")",
            "base_model=\(metadata["base_model"] ?? "")",
            "dataset_fingerprint=\(metadata["dataset_fingerprint"] ?? "")",
            "max_sequence_length=\(metadata["max_sequence_length"] ?? "")",
            "reasoning_effort=\(metadata["reasoning_effort"] ?? "")",
            "training_steps=\(config.trainingSteps)",
            "batch_size=\(config.batchSize)",
            "learning_rate=\(config.learningRate)",
            "weight_decay=\(config.weightDecay)",
            "beta1=\(config.beta1)",
            "beta2=\(config.beta2)",
            "epsilon=\(config.epsilon)",
            "seed=\(config.seed)",
            "lora_rank=\(firstLayer?.loraRank ?? 0)",
            "lora_alpha=\(firstLayer?.loraAlpha ?? 0)",
            "layer_paths=\(loraLayers.keys.sorted().joined(separator: ","))",
        ]
        return LoRATrainingFingerprint.sha256Hex(fields.joined(separator: "\n"))
    }

    public static func checkpointMetadata(
        base metadata: [String: String],
        config: TextLoRATrainingConfig,
        loraLayers: [String: TrainableLoRALayer],
        completedSteps: Int
    ) -> [String: String] {
        var checkpointMetadata = metadata
        checkpointMetadata["checkpoint_schema"] = checkpointSchema
        checkpointMetadata["completed_steps"] = String(completedSteps)
        checkpointMetadata["training_steps"] = String(config.trainingSteps)
        checkpointMetadata["batch_size"] = String(config.batchSize)
        checkpointMetadata["learning_rate"] = String(config.learningRate)
        checkpointMetadata["weight_decay"] = String(config.weightDecay)
        checkpointMetadata["beta1"] = String(config.beta1)
        checkpointMetadata["beta2"] = String(config.beta2)
        checkpointMetadata["epsilon"] = String(config.epsilon)
        checkpointMetadata["seed"] = String(config.seed)
        checkpointMetadata["training_config_fingerprint"] = configFingerprint(
            config: config,
            loraLayers: loraLayers,
            metadata: metadata
        )
        return checkpointMetadata
    }

    private static func validateMetadata(
        _ metadata: [String: String]?,
        expected: [String: String]
    ) throws {
        let identityKeys = [
            "format",
            "base_model",
            "dataset_fingerprint",
            "max_sequence_length",
            "reasoning_effort",
        ]
        for key in identityKeys {
            guard let expectedValue = expected[key], !expectedValue.isEmpty else { continue }
            guard let actualValue = metadata?[key] else {
                throw TextLoRACheckpointError.missingMetadata(key)
            }
            guard actualValue == expectedValue else {
                throw TextLoRACheckpointError.metadataMismatch(
                    key: key,
                    expected: expectedValue,
                    actual: actualValue
                )
            }
        }
    }

    private static func validateRankAndAlpha(
        _ metadata: [String: String]?,
        loraLayers: [String: TrainableLoRALayer]
    ) throws {
        guard let firstLayer = loraLayers.sorted(by: { $0.key < $1.key }).first?.value else {
            throw TextLoRACheckpointError.noLoRALayers
        }
        guard let rankValue = metadata?["lora_rank"], let rank = Int(rankValue) else {
            throw TextLoRACheckpointError.missingMetadata("lora_rank")
        }
        guard rank == firstLayer.loraRank else {
            throw TextLoRACheckpointError.metadataMismatch(
                key: "lora_rank",
                expected: String(firstLayer.loraRank),
                actual: rankValue
            )
        }
        guard let alphaValue = metadata?["lora_alpha"], let alpha = Float(alphaValue) else {
            throw TextLoRACheckpointError.missingMetadata("lora_alpha")
        }
        guard alpha == firstLayer.loraAlpha else {
            throw TextLoRACheckpointError.metadataMismatch(
                key: "lora_alpha",
                expected: String(firstLayer.loraAlpha),
                actual: alphaValue
            )
        }
    }

    private static func resolveCompletedSteps(
        metadata: [String: String]?,
        sidecar: LoRATrainingCheckpointState?,
        override: Int?
    ) throws -> Int {
        let embeddedStep: Int?
        if let sidecar {
            embeddedStep = sidecar.step
        } else if let raw = metadata?["completed_steps"] {
            guard let parsed = Int(raw) else {
                throw TextLoRACheckpointError.invalidCompletedSteps(raw)
            }
            embeddedStep = parsed
        } else {
            embeddedStep = nil
        }

        if let embeddedStep, let override, embeddedStep != override {
            throw TextLoRACheckpointError.resumeStepMismatch(
                checkpoint: embeddedStep,
                requested: override
            )
        }
        guard let resolved = embeddedStep ?? override else {
            throw TextLoRACheckpointError.legacyResumeStepRequired
        }
        return resolved
    }

    private static func validateResumeState(
        _ sidecar: LoRATrainingCheckpointState?,
        metadata: [String: String]?,
        expectedMetadata: [String: String],
        expectedFingerprint: String,
        config: TextLoRATrainingConfig,
        completedSteps: Int
    ) throws {
        guard completedSteps > 0, completedSteps < config.trainingSteps else {
            throw TextLoRACheckpointError.completedStepsOutOfRange(
                completed: completedSteps,
                total: config.trainingSteps
            )
        }

        if let sidecar {
            guard sidecar.totalSteps == config.trainingSteps else {
                throw TextLoRACheckpointError.totalStepsMismatch(
                    checkpoint: sidecar.totalSteps,
                    requested: config.trainingSteps
                )
            }
            guard sidecar.seed == config.seed else {
                throw TextLoRACheckpointError.seedMismatch(
                    checkpoint: sidecar.seed,
                    requested: config.seed
                )
            }
            if let expected = expectedMetadata["format"], sidecar.format != expected {
                throw TextLoRACheckpointError.metadataMismatch(
                    key: "format",
                    expected: expected,
                    actual: sidecar.format
                )
            }
            if let expected = expectedMetadata["base_model"], sidecar.baseModel != expected {
                throw TextLoRACheckpointError.metadataMismatch(
                    key: "base_model",
                    expected: expected,
                    actual: sidecar.baseModel
                )
            }
            if let expected = expectedMetadata["dataset_fingerprint"],
               sidecar.datasetFingerprint != expected {
                throw TextLoRACheckpointError.metadataMismatch(
                    key: "dataset_fingerprint",
                    expected: expected,
                    actual: sidecar.datasetFingerprint ?? "<missing>"
                )
            }
            guard sidecar.configFingerprint == expectedFingerprint else {
                throw TextLoRACheckpointError.configFingerprintMismatch
            }
            return
        }

        if let schema = metadata?["checkpoint_schema"] {
            guard schema == checkpointSchema else {
                throw TextLoRACheckpointError.unsupportedSchema(schema)
            }
            guard metadata?["training_config_fingerprint"] == expectedFingerprint else {
                throw TextLoRACheckpointError.configFingerprintMismatch
            }
        } else if config.resumeStep == nil {
            throw TextLoRACheckpointError.legacyResumeStepRequired
        }
    }

    private static func tensorKeys(for path: String) -> [String] {
        [
            "\(path).lora_down.weight",
            "\(path).lora_up.weight",
            "\(path).lora_down.m",
            "\(path).lora_down.v",
            "\(path).lora_up.m",
            "\(path).lora_up.v",
        ]
    }

    private static func requiredArray(
        _ arrays: [String: MLXArray],
        key: String,
        shape: [Int]
    ) throws -> MLXArray {
        guard let array = arrays[key] else {
            throw TextLoRACheckpointError.tensorInventoryMismatch(missing: [key], unexpected: [])
        }
        guard array.shape == shape else {
            throw TextLoRACheckpointError.tensorShapeMismatch(
                key: key,
                expected: shape,
                actual: array.shape
            )
        }
        return array
    }

    private static func requiredOptimizerArray(
        _ arrays: [String: MLXArray],
        key: String,
        shape: [Int]
    ) throws -> MLXArray {
        let array = try requiredArray(arrays, key: key, shape: shape)
        guard array.dtype == .float32 else {
            throw TextLoRACheckpointError.optimizerStateDTypeMismatch(
                key: key,
                actual: String(describing: array.dtype)
            )
        }
        return array
    }
}

public enum TextLoRACheckpointError: Error, LocalizedError, Sendable {
    case fileNotFound(String)
    case optimizerStateRequired
    case noLoRALayers
    case missingMetadata(String)
    case metadataMismatch(key: String, expected: String, actual: String)
    case invalidCompletedSteps(String)
    case legacyResumeStepRequired
    case resumeStepMismatch(checkpoint: Int, requested: Int)
    case completedStepsOutOfRange(completed: Int, total: Int)
    case totalStepsMismatch(checkpoint: Int, requested: Int)
    case seedMismatch(checkpoint: UInt64, requested: UInt64)
    case configFingerprintMismatch
    case unsupportedSchema(String)
    case tensorInventoryMismatch(missing: [String], unexpected: [String])
    case tensorShapeMismatch(key: String, expected: [Int], actual: [Int])
    case optimizerStateDTypeMismatch(key: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Text LoRA resume checkpoint does not exist: \(path)"
        case .optimizerStateRequired:
            return "Text LoRA resume requires a checkpoint with Adam optimizer state."
        case .noLoRALayers:
            return "Text LoRA resume requires at least one injected LoRA layer."
        case .missingMetadata(let key):
            return "Text LoRA resume checkpoint is missing required metadata '\(key)'."
        case .metadataMismatch(let key, let expected, let actual):
            return "Text LoRA resume metadata mismatch for '\(key)': expected '\(expected)', got '\(actual)'."
        case .invalidCompletedSteps(let value):
            return "Text LoRA resume checkpoint has invalid completed_steps '\(value)'."
        case .legacyResumeStepRequired:
            return "Legacy text LoRA checkpoints require an explicit --resume-step."
        case .resumeStepMismatch(let checkpoint, let requested):
            return "Text LoRA resume step mismatch: checkpoint is at \(checkpoint), requested \(requested)."
        case .completedStepsOutOfRange(let completed, let total):
            return "Text LoRA resume step \(completed) must be greater than zero and below total steps \(total)."
        case .totalStepsMismatch(let checkpoint, let requested):
            return "Text LoRA resume total-step mismatch: checkpoint planned \(checkpoint), requested \(requested)."
        case .seedMismatch(let checkpoint, let requested):
            return "Text LoRA resume seed mismatch: checkpoint used \(checkpoint), requested \(requested)."
        case .configFingerprintMismatch:
            return "Text LoRA resume training configuration does not match the checkpoint."
        case .unsupportedSchema(let schema):
            return "Unsupported text LoRA checkpoint schema '\(schema)'."
        case .tensorInventoryMismatch(let missing, let unexpected):
            return "Text LoRA resume tensor inventory mismatch; missing=\(missing), unexpected=\(unexpected)."
        case .tensorShapeMismatch(let key, let expected, let actual):
            return "Text LoRA resume tensor '\(key)' has shape \(actual), expected \(expected)."
        case .optimizerStateDTypeMismatch(let key, let actual):
            return "Text LoRA resume optimizer tensor '\(key)' has data type \(actual); expected float32."
        }
    }
}
