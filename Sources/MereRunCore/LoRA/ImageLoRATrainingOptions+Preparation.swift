import Foundation

extension ImageLoRATrainingOptions {
    func prepareKrea(options: Resolved) throws -> ImageLoRATrainingPlan.Training {
        if options.checkpointInterval != nil {
            throw ImageLoRATrainingIssue("--checkpoint-interval is only supported for FLUX.2 Klein LoRA training")
        }
        if hasKleinOnlyTrainingOptions(options: options) {
            throw ImageLoRATrainingIssue("Klein training options require a FLUX.2 Klein base model.")
        }

        let examples: [Krea2LoRATrainingExample]
        let datasetRoot: String?
        if syntheticSamples != nil {
            examples = []
            datasetRoot = nil
        } else {
            guard let data else {
                throw ImageLoRATrainingIssue("--data is required unless --synthetic-samples is set")
            }
            let dataURL = URL(fileURLWithPath: data).standardizedFileURL
            let pairs = try DatasetLoader.loadImageCaptionPairs(
                from: dataURL,
                excludePreviewImages: excludePreviewImages
            )
            examples = pairs.map { pair in
                Krea2LoRATrainingExample(imageURL: pair.imageURL, caption: pair.caption)
            }
            datasetRoot = dataURL.path
        }

        var config = Krea2LoRATrainingConfig()
        config.width = options.width
        config.height = options.height
        config.maxTextLength = maxTextLength
        config.schedulerSteps = schedulerSteps
        config.trainingSteps = options.trainingSteps
        config.batchSize = batchSize
        config.learningRate = options.learningRate
        config.seed = seed
        config.loraRank = options.rank
        config.loraAlpha = options.alpha
        config.captionDropout = options.captionDropout
        config.loraTargetSuffixes = lite ? Krea2LoRAInjector.liteTargetSuffixes : nil
        config.baseQuantizationBits = baseQuantizationBits
        config.syntheticSampleCount = syntheticSamples
        config.datasetRoot = datasetRoot
        config.useCompile = !options.noCompile
        if let lrWarmupSteps = options.lrWarmupSteps {
            config.lrWarmupSteps = lrWarmupSteps
        }
        if let useCosineScheduler = options.useCosineScheduler {
            config.useCosineScheduler = useCosineScheduler
        }
        if let lrMinFactor = options.lrMinFactor {
            config.lrMinFactor = lrMinFactor
        }

        return .krea(examples: examples, configuration: config)
    }

    func prepareKlein(options: Resolved) throws -> ImageLoRATrainingPlan.Training {
        if baseQuantizationBits != nil {
            throw ImageLoRATrainingIssue("--base-quantization-bits is only supported for Krea 2 LoRA training")
        }
        if syntheticSamples != nil {
            throw ImageLoRATrainingIssue("--synthetic-samples is only supported for Krea 2 LoRA smoke tests")
        }
        guard let data else {
            throw ImageLoRATrainingIssue("--data is required")
        }

        let dataURL = URL(fileURLWithPath: data).standardizedFileURL
        let pairs = try DatasetLoader.loadImageCaptionPairs(
            from: dataURL,
            excludePreviewImages: excludePreviewImages
        )
        let examples = pairs.map { pair in
            Flux2KleinLoRATrainingExample(imageURL: pair.imageURL, caption: pair.caption)
        }

        var config = Flux2KleinLoRATrainingConfig()
        let rankPreset = try Self.resolveKleinRankPreset(loraRankPreset)
        let resolvedRank = rankPreset?.rank ?? options.rank
        let targetRanks = try Self.resolveKleinTargetPreset(options.loraTargetPreset, rank: resolvedRank)
        let targetRankSuffixes = try loraTargetRanks.map(Self.parseKleinTargetRankSuffixes) ?? rankPreset?.targetRankSuffixes
        config.width = options.width
        config.height = options.height
        config.maxResolution = options.maxResolution
        config.maxTextLength = maxTextLength
        config.schedulerSteps = schedulerSteps
        config.trainingSteps = benchmarkSteps.map { benchmarkWarmupSteps + $0 } ?? options.trainingSteps
        config.batchSize = batchSize
        config.learningRate = options.learningRate
        config.seed = seed
        config.loraRank = resolvedRank
        config.loraAlpha = options.alpha ?? rankPreset?.alpha ?? Float(config.loraRank)
        config.loraTargetMode = try Self.resolveKleinLoRATargetMode(loraTargetMode)
        config.captionDropout = options.captionDropout
        config.loraTargetSuffixes = lite ? Self.kleinLiteTargetSuffixes : nil
        config.loraTargetRanks = targetRanks
        config.loraTargetRankSuffixes = targetRankSuffixes
        config.checkpointInterval = options.checkpointInterval
        config.sampleInterval = sampleInterval
        config.samplePrompt = samplePrompt
        config.progressive = progressive
        config.lowRam = options.lowRam
        config.gradientCheckpointing = gradientCheckpointing
        config.benchmarkSteps = benchmarkSteps
        config.benchmarkWarmupSteps = benchmarkWarmupSteps
        if options.noCompile || gradientCheckpointing {
            config.useCompile = false
        }
        config.datasetRoot = dataURL.path
        if let raw = timestepSampling {
            guard let parsed = Flux2TimestepSamplingStrategy(rawValue: raw) else {
                throw ImageLoRATrainingIssue("Unsupported --timestep-sampling '\(raw)'")
            }
            config.timestepSampling = parsed
        }
        if let raw = timestepLossWeighting {
            guard let parsed = Flux2TimestepLossWeightingStrategy(rawValue: raw) else {
                throw ImageLoRATrainingIssue("Unsupported --timestep-loss-weighting '\(raw)'")
            }
            config.timestepLossWeighting = parsed
        }
        if let raw = lossWeighting {
            guard let parsed = Flux2LossWeightingStrategy(rawValue: raw) else {
                throw ImageLoRATrainingIssue("Unsupported --loss-weighting '\(raw)'")
            }
            config.lossWeighting = parsed
        }
        if let timestepLow {
            config.timestepLow = timestepLow
        }
        if let timestepHigh {
            config.timestepHigh = timestepHigh
        }
        if let lrWarmupSteps {
            config.lrWarmupSteps = lrWarmupSteps
        }
        if noCosineScheduler {
            config.useCosineScheduler = false
        }
        if let lrMinFactor {
            config.lrMinFactor = lrMinFactor
        }
        if let adamWeightDecay {
            config.adamWeightDecay = adamWeightDecay
        }

        let sample = try prepareSample(fallbackPrompt: examples.first?.caption ?? "", options: options)
        return .klein(
            examples: examples, configuration: config,
            resumeFrom: resumeFrom.map { URL(fileURLWithPath: $0).standardizedFileURL }, sample: sample
        )
    }

    private func prepareSample(
        fallbackPrompt: String, options: Resolved
    ) throws -> ImageLoRATrainingPlan.Sample? {
        guard sampleInterval != nil else { return nil }
        let modelPath = try resolveKleinSampleModelPath()
        let prompt = (samplePrompt ?? fallbackPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw ImageLoRATrainingIssue("--sample-prompt is empty and the first caption is empty")
        }
        return .init(
            modelPath: modelPath, prompt: prompt, width: options.width, height: options.height,
            steps: sampleSteps, guidanceScale: sampleGuidanceScale, loraScale: sampleLoRAScale,
            seed: sampleSeed ?? (seed == 0 ? 42 : seed)
        )
    }
}
