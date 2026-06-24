import Foundation
import MLX
import MLXNN
import MLXRandom

public enum Krea2LoRATrainerError: Error, LocalizedError {
    case datasetEmpty
    case invalidDimensions(width: Int, height: Int)
    case invalidTrainingSteps(Int)
    case invalidBatchSize(Int)
    case invalidSchedulerSteps(Int)
    case outputMustBeSafetensors(URL)
    case imageNotFound(URL)
    case captionEmpty(URL)
    case imageDecodeFailed(URL)
    case modelPathNotFound(String)
    case baseModelNotTrainable(String)

    public var errorDescription: String? {
        switch self {
        case .datasetEmpty:
            return "Training dataset is empty."
        case .invalidDimensions(let width, let height):
            return "Invalid image dimensions (\(width)x\(height)); width/height must be >0 and divisible by 16."
        case .invalidTrainingSteps(let steps):
            return "Training steps must be >= 1 (got \(steps))."
        case .invalidBatchSize(let batchSize):
            return "Batch size must be >= 1 (got \(batchSize))."
        case .invalidSchedulerSteps(let steps):
            return "Scheduler steps must be >= 1 (got \(steps))."
        case .outputMustBeSafetensors(let url):
            return "Output must be a .safetensors file: \(url.path)"
        case .imageNotFound(let url):
            return "Image not found: \(url.path)"
        case .captionEmpty(let url):
            return "Caption is empty for: \(url.path)"
        case .imageDecodeFailed(let url):
            return "Failed to decode image: \(url.path)"
        case .modelPathNotFound(let path):
            return "Model path not found: \(path)"
        case .baseModelNotTrainable(let id):
            return "Krea 2 LoRA training requires the Raw/base checkpoint; got \(id)."
        }
    }
}

public struct Krea2LoRATrainingExample: Hashable, Sendable {
    public let imageURL: URL
    public let caption: String

    public init(imageURL: URL, caption: String) {
        self.imageURL = imageURL
        self.caption = caption
    }
}

public struct Krea2LoRATrainingConfig: Sendable {
    public var width: Int
    public var height: Int
    public var maxTextLength: Int
    public var schedulerSteps: Int
    public var trainingSteps: Int
    public var batchSize: Int
    public var learningRate: Float
    public var seed: UInt64
    public var loraRank: Int
    public var loraAlpha: Float?
    public var loraTargetPrefixes: [String]?
    public var loraTargetSuffixes: [String]?
    public var captionDropout: Float
    public var saveDType: DType
    public var logEvery: Int
    public var baseModelId: String
    public var syntheticSampleCount: Int?
    public var timestepLow: Int
    public var timestepHigh: Int?
    public var datasetRoot: String?

    public init(
        width: Int = 1024,
        height: Int = 1024,
        maxTextLength: Int = 512,
        schedulerSteps: Int = 1000,
        trainingSteps: Int = 1000,
        batchSize: Int = 1,
        learningRate: Float = 1e-4,
        seed: UInt64 = 0,
        loraRank: Int = 16,
        loraAlpha: Float? = nil,
        loraTargetPrefixes: [String]? = nil,
        loraTargetSuffixes: [String]? = nil,
        captionDropout: Float = 0.05,
        saveDType: DType = .float16,
        logEvery: Int = 10,
        baseModelId: String = Krea2RawResources.modelId,
        syntheticSampleCount: Int? = nil,
        timestepLow: Int = 0,
        timestepHigh: Int? = nil,
        datasetRoot: String? = nil
    ) {
        self.width = width
        self.height = height
        self.maxTextLength = maxTextLength
        self.schedulerSteps = schedulerSteps
        self.trainingSteps = trainingSteps
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.seed = seed
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha
        self.loraTargetPrefixes = loraTargetPrefixes
        self.loraTargetSuffixes = loraTargetSuffixes
        self.captionDropout = captionDropout
        self.saveDType = saveDType
        self.logEvery = logEvery
        self.baseModelId = baseModelId
        self.syntheticSampleCount = syntheticSampleCount
        self.timestepLow = timestepLow
        self.timestepHigh = timestepHigh
        self.datasetRoot = datasetRoot
    }
}

public struct Krea2LoRATrainingProgress: Sendable {
    public enum Stage: Sendable {
        case loadingModels
        case encodingDataset(current: Int, total: Int)
        case injectingLoRA(layerCount: Int)
        case training(step: Int, total: Int, loss: Float?)
        case saving
    }

    public let stage: Stage
    public let fraction: Float

    public init(stage: Stage, fraction: Float) {
        self.stage = stage
        self.fraction = fraction
    }
}

public enum Krea2LoRATrainer {
    public static func train(
        modelPath: String,
        examples: [Krea2LoRATrainingExample],
        outputURL: URL,
        config: Krea2LoRATrainingConfig = Krea2LoRATrainingConfig(),
        progressHandler: (@Sendable (Krea2LoRATrainingProgress) -> Void)? = nil
    ) async throws {
        guard !examples.isEmpty || (config.syntheticSampleCount ?? 0) > 0 else {
            throw Krea2LoRATrainerError.datasetEmpty
        }
        guard config.width > 0, config.height > 0, config.width % 16 == 0, config.height % 16 == 0 else {
            throw Krea2LoRATrainerError.invalidDimensions(width: config.width, height: config.height)
        }
        guard config.trainingSteps >= 1 else {
            throw Krea2LoRATrainerError.invalidTrainingSteps(config.trainingSteps)
        }
        guard config.batchSize >= 1 else {
            throw Krea2LoRATrainerError.invalidBatchSize(config.batchSize)
        }
        guard config.schedulerSteps >= 1 else {
            throw Krea2LoRATrainerError.invalidSchedulerSteps(config.schedulerSteps)
        }
        guard outputURL.pathExtension.lowercased() == "safetensors" else {
            throw Krea2LoRATrainerError.outputMustBeSafetensors(outputURL)
        }

        let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw Krea2LoRATrainerError.modelPathNotFound(modelPath)
        }

        let modelManifest = try MereRunModelManifest.loadRequired(from: modelURL)
        guard modelManifest.engine == .krea2, modelManifest.variant == .base else {
            throw Krea2LoRATrainerError.baseModelNotTrainable(modelManifest.id)
        }

        progressHandler?(Krea2LoRATrainingProgress(stage: .loadingModels, fraction: 0))
        let resources = Krea2Resources(rootURL: modelURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Krea2GeneratorError.missingModelFiles(missing)
        }

        let configs = try Krea2ModelConfigs.load(from: resources)
        let maxTextLength = min(config.maxTextLength, 512)
        let tokenizer = try QwenTokenizer.load(from: resources.tokenizerURL, maxLengthOverride: maxTextLength)
        let textEncoder = try Krea2ModelLoader.loadTextEncoder(
            from: resources,
            configuration: configs.textEncoder
        )
        let transformer = try Krea2ModelLoader.loadTransformer(
            from: resources,
            configuration: configs.transformer
        )
        let vae = try Krea2ModelLoader.loadVAE(from: resources, configuration: configs.vae)
        MLX.eval(textEncoder, transformer, vae)
        progressHandler?(Krea2LoRATrainingProgress(stage: .loadingModels, fraction: 1))

        struct PreparedExample {
            let imageTokens: MLXArray
            let textHiddenStates: MLXArray
            let textMask: MLXArray
            let positionIds: MLXArray
            let validMask: MLXArray
        }

        let datasetFingerprint = LoRATrainingFingerprint.sha256Hex(
            examples
                .map { [$0.imageURL.standardizedFileURL.path, $0.caption].joined(separator: "|") }
                .joined(separator: "\n")
        )
        let runDataFingerprint = makeRunDataFingerprint(examples: examples, dataRootPath: config.datasetRoot)
        let serializedTargetSuffixes = (config.loraTargetSuffixes ?? Krea2LoRAInjector.defaultTargetSuffixes)
            .joined(separator: ",")
        let serializedTargetPrefixes = (config.loraTargetPrefixes ?? Krea2LoRAInjector.defaultTargetPrefixes)
            .joined(separator: ",")
        let configFingerprintInput = [
            "model:\(modelURL.path)",
            "model_id:\(config.baseModelId)",
            "size:\(config.width)x\(config.height)",
            "scheduler_steps:\(config.schedulerSteps)",
            "training_steps:\(config.trainingSteps)",
            "batch_size:\(config.batchSize)",
            "learning_rate:\(config.learningRate)",
            "rank:\(config.loraRank)",
            "alpha:\(config.loraAlpha.map { "\($0)" } ?? "")",
            "max_text_length:\(maxTextLength)",
            "caption_dropout:\(config.captionDropout)",
            "timestep_low:\(config.timestepLow)",
            "timestep_high:\(config.timestepHigh.map { "\($0)" } ?? "")",
            "synthetic_sample_count:\(config.syntheticSampleCount.map { "\($0)" } ?? "")",
            "lora_target_prefixes:\(serializedTargetPrefixes)",
            "lora_target_suffixes:\(serializedTargetSuffixes)",
        ].joined(separator: "\n")
        let configFingerprint = LoRATrainingFingerprint.sha256Hex(configFingerprintInput)

        let latentHeight = config.height / Krea2SampleBuilder.vaeCompression
        let latentWidth = config.width / Krea2SampleBuilder.vaeCompression
        let imageTokenHeight = latentHeight / configs.transformer.patchSize
        let imageTokenWidth = latentWidth / configs.transformer.patchSize
        let imageTokenCount = imageTokenHeight * imageTokenWidth
        var prepared: [PreparedExample] = []
        let syntheticCount = config.syntheticSampleCount ?? 0

        if syntheticCount > 0 {
            prepared.reserveCapacity(syntheticCount)
            for _ in 0..<syntheticCount {
                let textLength = max(1, maxTextLength - 34)
                let cleanLatents = MLXRandom.normal([
                    1,
                    configs.transformer.latentChannels,
                    latentHeight,
                    latentWidth,
                ]).asType(.bfloat16)
                let textHidden = MLXRandom.normal([
                    1,
                    textLength,
                    configs.transformer.numTextLayers,
                    configs.transformer.textHiddenDim,
                ]).asType(.bfloat16)
                let textMask = MLXArray.ones([1, textLength], dtype: .int32)
                let preparedSample = Krea2SampleBuilder.prepare(
                    latents: cleanLatents,
                    textLength: textLength,
                    textMask: textMask,
                    patch: configs.transformer.patchSize
                )
                let validMask = MLX.concatenated([
                    textMask,
                    MLXArray.ones([1, imageTokenCount], dtype: .int32),
                ], axis: 1)
                prepared.append(
                    PreparedExample(
                        imageTokens: preparedSample.imageTokens.asType(.bfloat16),
                        textHiddenStates: textHidden,
                        textMask: textMask,
                        positionIds: preparedSample.positionIds,
                        validMask: validMask
                    )
                )
            }
        } else {
            prepared.reserveCapacity(examples.count)
            for (index, example) in examples.enumerated() {
                try Task.checkCancellation()
                progressHandler?(Krea2LoRATrainingProgress(
                    stage: .encodingDataset(current: index, total: examples.count),
                    fraction: Float(index) / Float(max(examples.count, 1))
                ))
                let caption = example.caption.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !caption.isEmpty else {
                    throw Krea2LoRATrainerError.captionEmpty(example.imageURL)
                }

                let text = try encodePrompt(
                    caption,
                    tokenizer: tokenizer,
                    encoder: textEncoder,
                    selectedLayers: configs.selectedTextLayers,
                    maxLength: maxTextLength
                )
                let cleanLatents = try encodeTrainingImage(
                    example.imageURL,
                    vae: vae,
                    config: configs.vae,
                    width: config.width,
                    height: config.height
                )
                let preparedSample = Krea2SampleBuilder.prepare(
                    latents: cleanLatents,
                    textLength: text.hiddenStates.dim(1),
                    textMask: text.attentionMask,
                    patch: configs.transformer.patchSize
                )
                prepared.append(
                    PreparedExample(
                        imageTokens: preparedSample.imageTokens.asType(.bfloat16),
                        textHiddenStates: text.hiddenStates.asType(.bfloat16),
                        textMask: text.attentionMask,
                        positionIds: preparedSample.positionIds,
                        validMask: preparedSample.validMask
                    )
                )
                MLX.eval(prepared.last!.imageTokens, prepared.last!.textHiddenStates)
            }
        }

        progressHandler?(Krea2LoRATrainingProgress(
            stage: .encodingDataset(current: prepared.count, total: prepared.count),
            fraction: 1
        ))

        let emptyPrompt: (hiddenStates: MLXArray, attentionMask: MLXArray)? = try {
            guard config.captionDropout > 0 else { return nil }
            if syntheticCount > 0 {
                let sample = prepared[0]
                return (
                    MLXArray.zeros(sample.textHiddenStates.shape, dtype: .bfloat16),
                    MLXArray.zeros(sample.textMask.shape, dtype: .int32)
                )
            }
            return try encodePrompt(
                "",
                tokenizer: tokenizer,
                encoder: textEncoder,
                selectedLayers: configs.selectedTextLayers,
                maxLength: maxTextLength
            )
        }()

        let effectiveAlpha = config.loraAlpha ?? Float(config.loraRank)
        let loraLayers = try Krea2LoRAInjector.inject(
            into: transformer,
            rank: config.loraRank,
            alpha: effectiveAlpha,
            targetPrefixes: config.loraTargetPrefixes ?? Krea2LoRAInjector.defaultTargetPrefixes,
            targetSuffixes: config.loraTargetSuffixes ?? Krea2LoRAInjector.defaultTargetSuffixes,
            zeroInitUp: true
        )
        try transformer.freeze(recursive: true, keys: nil, strict: false)
        for layer in loraLayers.values {
            guard let module = layer as? Module else { continue }
            try module.unfreeze(recursive: false, keys: ["loraDown", "loraUp"], strict: true)
        }
        ZImageTurboLoRATrainer.initializeAdamStateIfNeeded(for: loraLayers)
        let loraState = ZImageTurboLoRATrainer.LoRAState(loraLayers: loraLayers)
        MLX.eval(transformer, loraState)
        progressHandler?(Krea2LoRATrainingProgress(stage: .injectingLoRA(layerCount: loraLayers.count), fraction: 1))

        let lossAndGrad = valueAndGrad(model: transformer) { model, arrays in
            let clean = arrays[0]
            let noise = arrays[1]
            let textHiddenStates = arrays[2]
            let timestep = arrays[3].asType(.float32)
            let positionIds = arrays[4]
            let validMask = arrays[5]
            let sigma = timestep.ndim == 0 ? timestep : timestep.expandedDimensions(axis: 1).expandedDimensions(axis: 2)

            let noisy = ((MLXArray(1.0) - sigma) * clean.asType(.float32) + sigma * noise.asType(.float32))
                .asType(.bfloat16)
            let pred = model(
                imageTokens: noisy,
                textContext: textHiddenStates,
                timestep: timestep.asType(.bfloat16),
                positionIds: positionIds,
                validMask: validMask
            )
            let target = noise.asType(.float32) - clean.asType(.float32)
            return [(pred.asType(.float32) - target).square().mean()]
        }

        let metricsLogger = try LoRATrainingMetricsLogger(baseOutputURL: outputURL, resumeExisting: false)
        let effectiveSeed = config.seed == 0 ? UInt64(Date().timeIntervalSince1970) : config.seed
        var rng = ZImageTurboLoRATrainer.SplitMix64(seed: effectiveSeed)
        let loraLayerList = loraState.layers
        let beta1 = MLXArray(Float(0.9))
        let beta2 = MLXArray(Float(0.999))
        let oneMinusBeta1 = MLXArray(Float(0.1))
        let oneMinusBeta2 = MLXArray(Float(0.001))
        let eps = MLXArray(Float(1e-8))
        let lr = MLXArray(config.learningRate)
        let oneMinusLrWd = MLXArray(1.0 - config.learningRate * 0.0001)
        let timestepHigh = config.timestepHigh ?? config.schedulerSteps
        let timestepLow = config.timestepLow

        let state: [any Updatable] = [loraState]
        let trainStep = compile(inputs: state, outputs: state) { inputs -> [MLXArray] in
            let (values, grads) = lossAndGrad(transformer, inputs)
            let gradMap = Dictionary(uniqueKeysWithValues: grads.flattened())
            ZImageTurboLoRATrainer.applyAdamW(
                loraLayers: loraLayerList,
                gradMap: gradMap,
                lr: lr,
                beta1: beta1,
                beta2: beta2,
                oneMinusBeta1: oneMinusBeta1,
                oneMinusBeta2: oneMinusBeta2,
                eps: eps,
                oneMinusLrWd: oneMinusLrWd,
                useWeightDecay: true
            )
            return values
        }

        for step in 0..<config.trainingSteps {
            try Task.checkCancellation()
            let batchSize = min(config.batchSize, prepared.count)
            var cleanParts: [MLXArray] = []
            var noiseParts: [MLXArray] = []
            var textParts: [MLXArray] = []
            var positionParts: [MLXArray] = []
            var maskParts: [MLXArray] = []
            var timestepValues: [Float] = []

            for _ in 0..<batchSize {
                let index = Int(rng.next() % UInt64(prepared.count))
                let item = prepared[index]
                cleanParts.append(item.imageTokens)
                let useEmptyPrompt = emptyPrompt != nil
                    && Float(rng.next() % 1000) / 1000.0 < config.captionDropout
                if useEmptyPrompt, let emptyPrompt {
                    textParts.append(emptyPrompt.hiddenStates.asType(.bfloat16))
                    let validMask = MLX.concatenated([
                        emptyPrompt.attentionMask,
                        MLXArray.ones([1, imageTokenCount], dtype: .int32),
                    ], axis: 1)
                    maskParts.append(validMask)
                } else {
                    textParts.append(item.textHiddenStates)
                    maskParts.append(item.validMask)
                }
                positionParts.append(item.positionIds)
                let noiseSeed = UInt64(UInt32(truncatingIfNeeded: rng.next()))
                noiseParts.append(
                    MLXRandom.normal(item.imageTokens.shape, key: MLXRandom.key(noiseSeed))
                        .asType(.bfloat16)
                )
                let rawTimestep = Int(rng.next() % UInt64(max(timestepHigh - timestepLow, 1))) + timestepLow
                let clamped = min(max(rawTimestep, 0), config.schedulerSteps - 1)
                timestepValues.append(Float(clamped) / Float(config.schedulerSteps))
            }

            let cleanBatch = MLX.concatenated(cleanParts, axis: 0)
            let noiseBatch = MLX.concatenated(noiseParts, axis: 0)
            let textBatch = MLX.concatenated(textParts, axis: 0)
            let timestepBatch = MLXArray(timestepValues).asType(.float32)
            let positionBatch = MLX.concatenated(positionParts, axis: 0)
            let maskBatch = MLX.concatenated(maskParts, axis: 0)
            let loss = trainStep([cleanBatch, noiseBatch, textBatch, timestepBatch, positionBatch, maskBatch])[0]
            MLX.asyncEval(loss, loraState)

            let shouldLog = (step + 1) % max(config.logEvery, 1) == 0 || step == config.trainingSteps - 1
            let lossValue = shouldLog ? loss.item(Float.self) : nil
            if let lossValue {
                try metricsLogger.record(step: step + 1, loss: lossValue)
            }
            progressHandler?(Krea2LoRATrainingProgress(
                stage: .training(step: step + 1, total: config.trainingSteps, loss: lossValue),
                fraction: Float(step + 1) / Float(config.trainingSteps)
            ))
        }

        progressHandler?(Krea2LoRATrainingProgress(stage: .saving, fraction: 0))
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let metadata = [
            "format": "mererun.krea2.lora",
            "base_model": config.baseModelId,
            "recommended_runtime_model": Krea2Resources.modelId,
            "step": "\(config.trainingSteps)",
            "total_steps": "\(config.trainingSteps)",
            "seed": "\(effectiveSeed)",
            "dataset_fingerprint": datasetFingerprint,
            "config_fingerprint": configFingerprint,
        ]
        try LoRASafetensorsWriter.save(
            loraLayers: loraLayers,
            to: outputURL,
            dtype: config.saveDType,
            includeOptimizerState: true,
            metadata: metadata
        )

        let checkpointState = LoRATrainingCheckpointState(
            format: "mererun.krea2.lora",
            baseModel: config.baseModelId,
            checkpointFile: outputURL.lastPathComponent,
            step: config.trainingSteps,
            totalSteps: config.trainingSteps,
            seed: effectiveSeed,
            rngState: rng.rawState,
            datasetFingerprint: datasetFingerprint,
            configFingerprint: configFingerprint,
            phaseSchedule: [
                LoRATrainingCheckpointState.Phase(
                    width: config.width,
                    height: config.height,
                    steps: config.trainingSteps,
                    sampleCount: prepared.count
                ),
            ],
            phaseCursor: nil,
            configSnapshot: [
                "model": modelURL.path,
                "model_id": config.baseModelId,
                "size": "\(config.width)x\(config.height)",
                "scheduler_steps": "\(config.schedulerSteps)",
                "training_steps": "\(config.trainingSteps)",
                "batch_size": "\(config.batchSize)",
                "learning_rate": "\(config.learningRate)",
                "rank": "\(config.loraRank)",
                "alpha": config.loraAlpha.map { "\($0)" } ?? "",
                "max_text_length": "\(maxTextLength)",
                "caption_dropout": "\(config.captionDropout)",
                "lora_target_prefixes": serializedTargetPrefixes,
                "lora_target_suffixes": serializedTargetSuffixes,
                "config_fingerprint": configFingerprint,
            ],
            lossCSVFile: metricsLogger.csvURL.lastPathComponent,
            lossHTMLFile: metricsLogger.htmlURL.lastPathComponent,
            manifestFile: LoRATrainingManifest.url(nextTo: outputURL).lastPathComponent
        )
        try checkpointState.write(nextTo: outputURL)

        let manifest = LoRATrainingManifest(
            format: "mererun.krea2.lora",
            baseModel: config.baseModelId,
            outputFile: outputURL.lastPathComponent,
            emaOutputFile: nil,
            training: LoRATrainingManifest.Training(
                width: config.width,
                height: config.height,
                trainingSteps: config.trainingSteps,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: effectiveSeed,
                datasetCount: examples.isEmpty ? prepared.count : examples.count,
                checkpointInterval: nil,
                sampleInterval: nil,
                samplePrompt: nil,
                emaDecay: 0
            ),
            lora: LoRATrainingManifest.LoRA(
                rank: config.loraRank,
                alpha: effectiveAlpha,
                saveDType: String(describing: config.saveDType),
                includesOptimizerState: true
            ),
            extras: [
                "recommended_runtime_model": Krea2Resources.modelId,
                "max_text_length": "\(maxTextLength)",
                "scheduler_steps": "\(config.schedulerSteps)",
                "caption_dropout": "\(config.captionDropout)",
                "timestep_low": "\(config.timestepLow)",
                "timestep_high": config.timestepHigh.map { "\($0)" } ?? "",
                "synthetic_sample_count": config.syntheticSampleCount.map { "\($0)" } ?? "",
                "lora_target_prefixes": serializedTargetPrefixes,
                "lora_target_suffixes": serializedTargetSuffixes,
                "dataset_root": config.datasetRoot ?? "",
                "seed": "\(effectiveSeed)",
                "rng_state": "\(rng.rawState)",
                "dataset_fingerprint": datasetFingerprint,
                "config_fingerprint": configFingerprint,
                "checkpoint_sidecar_file": LoRATrainingCheckpointState.url(nextTo: outputURL).lastPathComponent,
                "loss_csv_file": metricsLogger.csvURL.lastPathComponent,
                "loss_html_file": metricsLogger.htmlURL.lastPathComponent,
            ]
        )
        try manifest.write(nextTo: outputURL)

        let runManifest = LoRATrainingRunManifest(
            format: "mererun.krea2.lora",
            model: config.baseModelId,
            isEdit: false,
            dataRoot: config.datasetRoot,
            dataRootRelative: LoRATrainingRunManifest.relativePath(
                from: outputURL.deletingLastPathComponent(),
                to: config.datasetRoot
            ),
            dataFingerprint: runDataFingerprint,
            checkpointFiles: [
                "lora_adapter": outputURL.lastPathComponent,
                "checkpoint_state": LoRATrainingCheckpointState.url(nextTo: outputURL).lastPathComponent,
                "manifest": LoRATrainingManifest.url(nextTo: outputURL).lastPathComponent,
                "loss_csv": metricsLogger.csvURL.lastPathComponent,
                "loss_html": metricsLogger.htmlURL.lastPathComponent,
                "optimizer": outputURL.lastPathComponent,
            ],
            step: config.trainingSteps,
            totalSteps: config.trainingSteps,
            seed: effectiveSeed,
            rngState: rng.rawState,
            datasetFingerprint: datasetFingerprint,
            configFingerprint: configFingerprint,
            phaseSchedule: checkpointState.phaseSchedule,
            phaseCursor: nil,
            configSnapshot: checkpointState.configSnapshot
        )
        try runManifest.write(nextTo: outputURL)
        try metricsLogger.writeArtifacts()
        _ = try? LoRACheckpointArchive.createZipBundle(
            primaryFile: outputURL,
            additionalFiles: [
                LoRATrainingCheckpointState.url(nextTo: outputURL),
                LoRATrainingManifest.url(nextTo: outputURL),
                LoRATrainingRunManifest.url(nextTo: outputURL),
                metricsLogger.csvURL,
                metricsLogger.htmlURL,
            ]
        )
        progressHandler?(Krea2LoRATrainingProgress(stage: .saving, fraction: 1))
    }

    static func encodePrompt(
        _ prompt: String,
        tokenizer: QwenTokenizer,
        encoder: QwenEncoder,
        selectedLayers: [Int],
        maxLength: Int
    ) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        let prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        let suffix = "<|im_end|>\n<|im_start|>assistant\n"
        let prefixDropCount = 34
        let suffixIds = tokenizer.encodeText(suffix)
        let inputs = Krea2SampleBuilder.paddedTextTokenInputs(
            promptTokenIds: tokenizer.encodeText(prefix + prompt),
            suffixTokenIds: suffixIds,
            padTokenId: tokenizer.padTokenId,
            maxLength: maxLength,
            prefixDropCount: prefixDropCount
        )
        let tokenIds = inputs.tokenIds
        let inputIds = MLXArray(tokenIds.map(Int32.init)).reshaped(1, tokenIds.count)
        let attentionMask = MLXArray(inputs.attentionMask).reshaped(1, inputs.attentionMask.count)
        let hiddenStates = encoder.forwardActivationHiddenStates(
            inputIds: inputIds,
            attentionMask: attentionMask,
            activationLayers: Krea2SampleBuilder.qwenActivationLayerIndices(from: selectedLayers)
        )
        let stacked = MLX.stacked(hiddenStates, axis: 2).asType(.bfloat16)
        let croppedStates = stacked[0..., prefixDropCount..., 0..., 0...]
        let croppedMask = attentionMask[0..., prefixDropCount...]
        MLX.eval(croppedStates, croppedMask)
        return (croppedStates, croppedMask)
    }

    static func encodeTrainingImage(
        _ url: URL,
        vae: QwenImageEditVAE,
        config: QwenImageEditVAEConfig,
        width: Int,
        height: Int
    ) throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Krea2LoRATrainerError.imageNotFound(url)
        }

        let resized = try QwenImageIO.resizedCenterCropPixelArray(
            from: url,
            width: width,
            height: height,
            addBatchDimension: true,
            dtype: .float32
        )
        let normalized = QwenImageIO.normalizeForEncoder(resized)
        let encoded = vae.encode(normalized).asType(.float32)

        var modelLatents = encoded / MLXArray(config.scalingFactor)
        if let shift = config.shiftFactor, shift != 0 {
            modelLatents = modelLatents + MLXArray(shift)
        }
        if let mean = config.latentsMean, let std = config.latentsStd {
            let meanTensor = MLXArray(mean).reshaped(1, mean.count, 1, 1)
            let stdTensor = MLXArray(std).reshaped(1, std.count, 1, 1)
            modelLatents = (modelLatents - meanTensor) / stdTensor
        }
        return modelLatents.asType(.bfloat16)
    }

    private static func makeRunDataFingerprint(
        examples: [Krea2LoRATrainingExample],
        dataRootPath: String?
    ) -> LoRATrainingRunManifest.DataFingerprint {
        let imagePaths = examples.map { example in
            LoRATrainingRunManifest.dataPath(
                for: example.imageURL,
                dataRootPath: dataRootPath
            )
        }
        return LoRATrainingRunManifest.DataFingerprint(
            count: examples.count,
            images: imagePaths,
            inputImages: [],
            isEdit: false
        )
    }
}
