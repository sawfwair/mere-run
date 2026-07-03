import Foundation
import MLX
import MLXNN

public struct TextLoRATrainingConfig: Sendable, Hashable {
    public let trainingSteps: Int
    public let batchSize: Int
    public let learningRate: Float
    public let weightDecay: Float
    public let beta1: Float
    public let beta2: Float
    public let epsilon: Float
    public let seed: UInt64

    public init(
        trainingSteps: Int,
        batchSize: Int,
        learningRate: Float,
        weightDecay: Float = 0.0,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        seed: UInt64 = 42
    ) {
        self.trainingSteps = trainingSteps
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.seed = seed
    }
}

public struct TextLoRATrainingReport: Sendable, Hashable, Codable {
    public let steps: Int
    public let initialLoss: Float?
    public let finalLoss: Float?
    public let layerCount: Int
    public let outputPath: String?
}

public struct TextLoRATrainingProgress: Sendable, Hashable {
    public enum Stage: Sendable, Hashable {
        case training(step: Int, total: Int, loss: Float)
        case saving
    }

    public let stage: Stage

    public var fraction: Float {
        switch stage {
        case .training(let step, let total, _):
            guard total > 0 else { return 0 }
            return min(max(Float(step) / Float(total), 0), 1)
        case .saving:
            return 1
        }
    }

    public init(stage: Stage) {
        self.stage = stage
    }
}

/// Environment-tunable behavior specific to the text SFT trainer. Shared
/// image/text knobs (cache limit, save cadence, sync eval, footprint) live in
/// `LoRATrainingEnvironment`.
enum TextLoRATrainingEnvironment {
    /// Gathered-loss path: project only loss-masked target positions through
    /// the lm_head and use logSumExp-minus-gather cross entropy. Identical
    /// gradients to the full-logits path; skips the `[B*T, 262k]` matmul and
    /// float32 chain over prompt/pad rows. MERERUN_TEXT_LORA_TRAIN_GATHERED_LOSS=0
    /// restores the legacy full-logits loss.
    static let gatheredLossEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_TEXT_LORA_TRAIN_GATHERED_LOSS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()

    /// Loss-readback cadence in steps. Between boundaries the step is
    /// scheduled with asyncEval and the loop continues without a GPU→CPU
    /// sync, so graph construction overlaps execution.
    /// MERERUN_TEXT_LORA_TRAIN_LOG_EVERY overrides; 1 restores the legacy
    /// per-step synchronous readback.
    static let logEvery: Int = {
        if let raw = ProcessInfo.processInfo.environment["MERERUN_TEXT_LORA_TRAIN_LOG_EVERY"],
           let value = Int(raw), value >= 1 {
            return value
        }
        return 10
    }()

    /// Buffer-cache cap for text training. The shared image default (16 GB)
    /// starves text step working sets: image steps run minutes so allocation
    /// churn amortizes, but a text step is seconds and a sub-working-set cap
    /// showed 2× step time at ~900-token sequences. 32 GB removed the churn
    /// while still bounding the footprint (uncapped, the cache balloons past
    /// 100 GB at long sequences). MERERUN_LORA_TRAIN_CACHE_LIMIT_GB still
    /// overrides for both trainers.
    static let cacheLimitGB: Int = {
        if let raw = ProcessInfo.processInfo.environment["MERERUN_LORA_TRAIN_CACHE_LIMIT_GB"],
           let value = Int(raw), value >= 0 {
            return value
        }
        return 32
    }()
}

public enum TextLoRATrainer {
    public static func train<Model: Module>(
        model: Model,
        loraLayers: [String: TrainableLoRALayer],
        examples: [TextSFTTokenizedExample],
        config: TextLoRATrainingConfig,
        outputURL: URL? = nil,
        metadata: [String: String] = [:],
        progressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)? = nil,
        gatheredForward: ((Model, MLXArray, MLXArray) -> MLXArray)? = nil,
        forward: @escaping (Model, MLXArray) -> MLXArray
    ) throws -> TextLoRATrainingReport {
        try validate(config: config, examples: examples, loraLayers: loraLayers)
        try model.freeze(recursive: true, keys: nil, strict: false)
        for layer in loraLayers.values {
            guard let module = layer as? Module else { continue }
            try module.unfreeze(recursive: false, keys: ["loraDown", "loraUp"], strict: true)
        }
        initializeAdamStateIfNeeded(for: loraLayers)

        let orderedLayers = loraLayers
            .map { (path: $0.key, layer: $0.value) }
            .sorted { $0.path < $1.path }

        let useGatheredLoss = gatheredForward != nil && TextLoRATrainingEnvironment.gatheredLossEnabled
        let lossAndGrad: (Model, [MLXArray]) -> ([MLXArray], ModuleParameters)
        if useGatheredLoss, let gatheredForward {
            lossAndGrad = valueAndGrad(model: model) { model, arrays in
                let logits = gatheredForward(model, arrays[0], arrays[1])
                return [TextSFTTrainingLoss.gatheredNextTokenCrossEntropy(logits: logits, labels: arrays[2])]
            }
        } else {
            lossAndGrad = valueAndGrad(model: model) { model, arrays in
                let logits = forward(model, arrays[0])
                let loss = TextSFTTrainingLoss.maskedNextTokenCrossEntropy(
                    logits: logits,
                    labels: arrays[1],
                    lossMask: arrays[2]
                )
                return [loss]
            }
        }
        if TextLoRATrainingEnvironment.cacheLimitGB > 0 {
            MLX.Memory.cacheLimit = TextLoRATrainingEnvironment.cacheLimitGB * 1_073_741_824
        }
        FileHandle.standardError.write(Data(
            "[text-lora-train] gathered_loss=\(useGatheredLoss) log_every=\(TextLoRATrainingEnvironment.logEvery) save_every=\(LoRATrainingEnvironment.periodicSaveInterval) cache_limit_gb=\(TextLoRATrainingEnvironment.cacheLimitGB)\n".utf8
        ))

        let beta1 = MLXArray(config.beta1)
        let beta2 = MLXArray(config.beta2)
        let oneMinusBeta1 = MLXArray(1 - config.beta1)
        let oneMinusBeta2 = MLXArray(1 - config.beta2)
        let eps = MLXArray(config.epsilon)
        let lr = MLXArray(config.learningRate)
        let oneMinusLrWd = MLXArray(1 - (config.learningRate * config.weightDecay))
        var initialLoss: Float?
        var finalLoss: Float?
        if let outputURL {
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        }
        let metricsLogger = try outputURL.map { try LoRATrainingMetricsLogger(baseOutputURL: $0, resumeExisting: false) }
        let trainingOrder = makeTrainingOrder(
            exampleCount: examples.count,
            drawCount: config.trainingSteps * config.batchSize,
            seed: config.seed
        )

        let logEvery = TextLoRATrainingEnvironment.logEvery
        let saveEvery = LoRATrainingEnvironment.periodicSaveInterval
        let partialURL = outputURL.map {
            $0.deletingPathExtension().appendingPathExtension("partial").appendingPathExtension("safetensors")
        }
        var lastBoundaryTime = Date()
        var lastBoundaryStep = 0

        for step in 0..<config.trainingSteps {
            let batchExamples = scheduledBatch(
                examples,
                order: trainingOrder,
                batchSize: config.batchSize,
                step: step
            )
            let stepInputs: [MLXArray]
            if useGatheredLoss {
                let batch = try TextSFTTrainingBatchBuilder.makeGatheredBatch(batchExamples)
                stepInputs = [batch.inputIds, batch.targetPositions, batch.targetLabels]
            } else {
                let batch = try TextSFTTrainingBatchBuilder.makeBatch(batchExamples)
                stepInputs = [batch.inputIds, batch.labels, batch.lossMask]
            }
            let (values, gradients) = lossAndGrad(model, stepInputs)
            let loss = values[0]
            let gradMap = Dictionary(uniqueKeysWithValues: gradients.flattened())
            let stepNumber = step + 1
            let biasCorrection1 = MLXArray(Float(1.0 - pow(Double(config.beta1), Double(stepNumber))))
            let biasCorrection2 = MLXArray(Float(1.0 - pow(Double(config.beta2), Double(stepNumber))))
            applyAdamW(
                loraLayers: orderedLayers,
                gradMap: gradMap,
                lr: lr,
                beta1: beta1,
                beta2: beta2,
                oneMinusBeta1: oneMinusBeta1,
                oneMinusBeta2: oneMinusBeta2,
                eps: eps,
                oneMinusLrWd: oneMinusLrWd,
                biasCorrection1: biasCorrection1,
                biasCorrection2: biasCorrection2,
                useWeightDecay: config.weightDecay > 0
            )
            let stepState = [loss] + orderedLayers.flatMap { [$0.layer.loraDown, $0.layer.loraUp] }
            let shouldSavePartial = saveEvery > 0 && stepNumber % saveEvery == 0
                && stepNumber < config.trainingSteps && partialURL != nil
            let isBoundary = stepNumber == 1
                || stepNumber == config.trainingSteps
                || stepNumber % logEvery == 0
                || shouldSavePartial
                || LoRATrainingEnvironment.synchronousStepEval
            guard isBoundary else {
                // Off-boundary steps are scheduled without a GPU→CPU sync so
                // the next step's graph construction overlaps execution;
                // asyncEval's back-pressure bounds how far the loop runs ahead.
                asyncEval(stepState)
                continue
            }
            eval(stepState)
            let scalarLoss = loss.item(Float.self)
            if initialLoss == nil { initialLoss = scalarLoss }
            finalLoss = scalarLoss
            let visibleStep = stepNumber
            let now = Date()
            let stepsCovered = max(visibleStep - lastBoundaryStep, 1)
            let secondsPerStep = now.timeIntervalSince(lastBoundaryTime) / Double(stepsCovered)
            lastBoundaryTime = now
            lastBoundaryStep = visibleStep
            let footprint = LoRATrainingEnvironment.currentPhysicalFootprintGB()
                .map { String(format: "%.1f", $0) } ?? "n/a"
            FileHandle.standardError.write(Data(
                "[text-lora-train] step=\(visibleStep)/\(config.trainingSteps) loss=\(scalarLoss) step_s=\(String(format: "%.2f", secondsPerStep)) footprint_gb=\(footprint)\n".utf8
            ))
            try metricsLogger?.record(step: visibleStep, loss: scalarLoss)
            progressHandler?(
                TextLoRATrainingProgress(
                    stage: .training(step: visibleStep, total: config.trainingSteps, loss: scalarLoss)
                )
            )
            if shouldSavePartial, let partialURL {
                try LoRASafetensorsWriter.save(
                    loraLayers: loraLayers,
                    to: partialURL,
                    includeOptimizerState: true,
                    metadata: metadata
                )
            }
        }

        progressHandler?(TextLoRATrainingProgress(stage: .saving))
        if let outputURL {
            try LoRASafetensorsWriter.save(
                loraLayers: loraLayers,
                to: outputURL,
                includeOptimizerState: true,
                metadata: metadata
            )
            try metricsLogger?.writeArtifacts()
            if let partialURL, FileManager.default.fileExists(atPath: partialURL.path) {
                try? FileManager.default.removeItem(at: partialURL)
            }
        }

        return TextLoRATrainingReport(
            steps: config.trainingSteps,
            initialLoss: initialLoss,
            finalLoss: finalLoss,
            layerCount: loraLayers.count,
            outputPath: outputURL?.path
        )
    }

    static func initializeAdamStateIfNeeded(for loraLayers: [String: TrainableLoRALayer]) {
        for layer in loraLayers.values {
            if layer.loraDownM == nil { layer.loraDownM = MLXArray.zeros(like: layer.loraDown) }
            if layer.loraDownV == nil { layer.loraDownV = MLXArray.zeros(like: layer.loraDown) }
            if layer.loraUpM == nil { layer.loraUpM = MLXArray.zeros(like: layer.loraUp) }
            if layer.loraUpV == nil { layer.loraUpV = MLXArray.zeros(like: layer.loraUp) }
        }
    }

    static func applyAdamW(
        loraLayers: [(path: String, layer: TrainableLoRALayer)],
        gradMap: [String: MLXArray],
        lr: MLXArray,
        beta1: MLXArray,
        beta2: MLXArray,
        oneMinusBeta1: MLXArray,
        oneMinusBeta2: MLXArray,
        eps: MLXArray,
        oneMinusLrWd: MLXArray,
        biasCorrection1: MLXArray,
        biasCorrection2: MLXArray,
        useWeightDecay: Bool
    ) {
        for (path, layer) in loraLayers {
            guard let downGrad = gradMap["\(path).loraDown"],
                  let upGrad = gradMap["\(path).loraUp"] else {
                continue
            }

            let mDown = beta1 * layer.loraDownM! + oneMinusBeta1 * downGrad
            let vDown = beta2 * layer.loraDownV! + oneMinusBeta2 * downGrad.square()
            let wDown = useWeightDecay ? layer.loraDown * oneMinusLrWd : layer.loraDown
            let mHatDown = mDown / biasCorrection1
            let vHatDown = vDown / biasCorrection2
            let newDown = wDown - lr * mHatDown / (vHatDown.sqrt() + eps)
            layer.loraDown._updateInternal(newDown)
            layer.loraDownM!._updateInternal(mDown)
            layer.loraDownV!._updateInternal(vDown)

            let mUp = beta1 * layer.loraUpM! + oneMinusBeta1 * upGrad
            let vUp = beta2 * layer.loraUpV! + oneMinusBeta2 * upGrad.square()
            let wUp = useWeightDecay ? layer.loraUp * oneMinusLrWd : layer.loraUp
            let mHatUp = mUp / biasCorrection1
            let vHatUp = vUp / biasCorrection2
            let newUp = wUp - lr * mHatUp / (vHatUp.sqrt() + eps)
            layer.loraUp._updateInternal(newUp)
            layer.loraUpM!._updateInternal(mUp)
            layer.loraUpV!._updateInternal(vUp)
        }
    }

    private static func validate(
        config: TextLoRATrainingConfig,
        examples: [TextSFTTokenizedExample],
        loraLayers: [String: TrainableLoRALayer]
    ) throws {
        guard config.trainingSteps >= 1 else {
            throw TextLoRATrainerError.invalidTrainingSteps(config.trainingSteps)
        }
        guard config.batchSize >= 1 else {
            throw TextLoRATrainerError.invalidBatchSize(config.batchSize)
        }
        guard config.learningRate > 0 else {
            throw TextLoRATrainerError.invalidLearningRate(config.learningRate)
        }
        guard !examples.isEmpty else {
            throw TextLoRATrainerError.emptyDataset
        }
        guard !loraLayers.isEmpty else {
            throw TextLoRATrainerError.noLoRALayers
        }
    }

    static func makeTrainingOrder(
        exampleCount: Int,
        drawCount: Int,
        seed: UInt64
    ) -> [Int] {
        guard exampleCount > 0, drawCount > 0 else { return [] }
        var order: [Int] = []
        order.reserveCapacity(drawCount)
        var epoch: UInt64 = 0
        while order.count < drawCount {
            var indices = Array(0..<exampleCount)
            var rng = SplitMix64(seed: seed &+ epoch &* 0x9E37_79B9_7F4A_7C15)
            if indices.count > 1 {
                for index in stride(from: indices.count - 1, through: 1, by: -1) {
                    let swapIndex = rng.nextInt(upperBound: index + 1)
                    indices.swapAt(index, swapIndex)
                }
            }
            order.append(contentsOf: indices.prefix(drawCount - order.count))
            epoch &+= 1
        }
        return order
    }

    private static func scheduledBatch(
        _ examples: [TextSFTTokenizedExample],
        order: [Int],
        batchSize: Int,
        step: Int
    ) -> [TextSFTTokenizedExample] {
        (0..<batchSize).map { offset in
            examples[order[step * batchSize + offset]]
        }
    }

    struct SplitMix64 {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }

        mutating func nextInt(upperBound: Int) -> Int {
            guard upperBound > 1 else { return 0 }
            return Int(next() % UInt64(upperBound))
        }
    }
}

public enum TextLoRATrainerError: Error, LocalizedError, Sendable {
    case invalidTrainingSteps(Int)
    case invalidBatchSize(Int)
    case invalidLearningRate(Float)
    case emptyDataset
    case noLoRALayers

    public var errorDescription: String? {
        switch self {
        case .invalidTrainingSteps(let value):
            return "Text LoRA training steps must be >= 1 (got \(value))."
        case .invalidBatchSize(let value):
            return "Text LoRA batch size must be >= 1 (got \(value))."
        case .invalidLearningRate(let value):
            return "Text LoRA learning rate must be > 0 (got \(value))."
        case .emptyDataset:
            return "Text LoRA training requires at least one tokenized example."
        case .noLoRALayers:
            return "Text LoRA training requires at least one injected LoRA layer."
        }
    }
}
