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

public enum TextLoRATrainer {
    public static func train<Model: Module>(
        model: Model,
        loraLayers: [String: TrainableLoRALayer],
        examples: [TextSFTTokenizedExample],
        config: TextLoRATrainingConfig,
        outputURL: URL? = nil,
        metadata: [String: String] = [:],
        progressHandler: (@Sendable (TextLoRATrainingProgress) -> Void)? = nil,
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

        let lossAndGrad = valueAndGrad(model: model) { model, arrays in
            let logits = forward(model, arrays[0])
            let loss = TextSFTTrainingLoss.maskedNextTokenCrossEntropy(
                logits: logits,
                labels: arrays[1],
                lossMask: arrays[2]
            )
            return [loss]
        }

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

        for step in 0..<config.trainingSteps {
            let batchExamples = scheduledBatch(
                examples,
                order: trainingOrder,
                batchSize: config.batchSize,
                step: step
            )
            let batch = try TextSFTTrainingBatchBuilder.makeBatch(batchExamples)
            let (values, gradients) = lossAndGrad(model, [batch.inputIds, batch.labels, batch.lossMask])
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
            eval([loss] + orderedLayers.flatMap { [$0.layer.loraDown, $0.layer.loraUp] })
            let scalarLoss = loss.item(Float.self)
            if initialLoss == nil { initialLoss = scalarLoss }
            finalLoss = scalarLoss
            let visibleStep = stepNumber
            try metricsLogger?.record(step: visibleStep, loss: scalarLoss)
            progressHandler?(
                TextLoRATrainingProgress(
                    stage: .training(step: visibleStep, total: config.trainingSteps, loss: scalarLoss)
                )
            )
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
