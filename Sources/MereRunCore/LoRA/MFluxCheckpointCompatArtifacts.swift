import Foundation

struct MFluxCheckpointCompatArtifacts: Sendable {
    let iteratorURL: URL
    let lossURL: URL
    let configURL: URL
    let checkpointManifestURL: URL

    var additionalFiles: [URL] {
        [iteratorURL, lossURL, configURL, checkpointManifestURL]
    }
}

enum MFluxCheckpointCompatArtifactsWriter {
    static func write(
        checkpointURL: URL,
        step: Int,
        seed: UInt64,
        batchSize: Int,
        datasetCount: Int,
        loraAdapterFileName: String,
        optimizerFileName: String?,
        iteratorCursor: MFluxResumeIteratorCompat.Cursor?,
        lossPoints: [LoRALossPoint],
        configSnapshot: [String: String]
    ) throws -> MFluxCheckpointCompatArtifacts {
        let checkpointDirectory = checkpointURL.deletingLastPathComponent()
        let safeStep = max(step, 0)
        let prefix = String(format: "%07d", safeStep)
        let iteratorURL = checkpointDirectory.appendingPathComponent("\(prefix)_iterator.json", isDirectory: false)
        let lossURL = checkpointDirectory.appendingPathComponent("\(prefix)_loss.json", isDirectory: false)
        let configURL = checkpointDirectory.appendingPathComponent("\(prefix)_config.json", isDirectory: false)
        let checkpointManifestURL = checkpointDirectory.appendingPathComponent("checkpoint.json", isDirectory: false)

        let iteratorPayload = MFluxResumeIteratorCompat.IteratorPayload(
            numIterations: safeStep,
            seed: seed,
            batchSize: max(batchSize, 1),
            currentPermutation: iteratorCursor?.permutation,
            position: iteratorCursor?.position,
            rngState: iteratorCursor?.pythonRNG?.serializedState(),
            localRNGState: iteratorCursor?.localRNGState
        )
        try writeJSON(iteratorPayload, to: iteratorURL)

        let sortedLossPoints = lossPoints.sorted { lhs, rhs in lhs.step < rhs.step }
        let lossPayload = sortedLossPoints.map { point in
            MFluxLossEntry(step: point.step, loss: Double(point.loss))
        }
        try writeJSON(lossPayload, to: lossURL)

        let configPayload = makeConfigPayload(
            snapshot: configSnapshot,
            checkpointDirectory: checkpointDirectory,
            step: safeStep,
            batchSize: batchSize,
            seed: seed
        )
        try writeJSON(configPayload, to: configURL)

        let resolvedOptimizer = (optimizerFileName ?? loraAdapterFileName)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let optimizerFile = resolvedOptimizer.isEmpty ? loraAdapterFileName : resolvedOptimizer
        let checkpointManifestPayload = MFluxCheckpointManifest(
            metadata: .init(numberOfTrainingData: max(datasetCount, 0)),
            files: [
                "config": configURL.lastPathComponent,
                "optimizer": optimizerFile,
                "lora_adapter": loraAdapterFileName,
                "iterator": iteratorURL.lastPathComponent,
                "loss": lossURL.lastPathComponent,
            ]
        )
        try writeJSON(checkpointManifestPayload, to: checkpointManifestURL)

        return MFluxCheckpointCompatArtifacts(
            iteratorURL: iteratorURL,
            lossURL: lossURL,
            configURL: configURL,
            checkpointManifestURL: checkpointManifestURL
        )
    }

    private static func makeConfigPayload(
        snapshot: [String: String],
        checkpointDirectory: URL,
        step: Int,
        batchSize: Int,
        seed: UInt64
    ) -> MFluxConfigPayload {
        let modelValue = normalizedValue(snapshot["model"]) ?? "unknown-model"
        let dataValue = normalizedValue(snapshot["dataset_root"]) ?? ""
        let schedulerSteps = intValue(snapshot["scheduler_steps"]) ?? 1000
        let learningRate = doubleValue(snapshot["learning_rate"]) ?? 1e-4
        let trainingLoop = MFluxTrainingLoopPayload(
            batchSize: max(batchSize, 1),
            numEpochs: 1,
            timestepLow: intValue(snapshot["timestep_low"]),
            timestepHigh: intValue(snapshot["timestep_high"])
        )
        let checkpointInterval = intValue(snapshot["checkpoint_interval"]) ?? max(step, 1)
        let targets = loraTargets(from: snapshot["lora_target_ranks"])
        return MFluxConfigPayload(
            model: modelValue,
            seed: seed,
            steps: max(schedulerSteps, 1),
            data: dataValue,
            trainingLoop: trainingLoop,
            optimizer: .init(name: "adamw", learningRate: learningRate),
            checkpoint: .init(saveFrequency: max(checkpointInterval, 1), outputPath: checkpointDirectory.path),
            maxResolution: intValue(snapshot["max_resolution"]).flatMap { $0 > 0 ? $0 : nil },
            loraLayers: targets.isEmpty ? nil : .init(targets: targets)
        )
    }

    private static func loraTargets(from serializedRanks: String?) -> [MFluxLoRATargetPayload] {
        guard let serializedRanks = normalizedValue(serializedRanks) else {
            return []
        }

        var targets: [MFluxLoRATargetPayload] = []
        for part in serializedRanks.split(separator: ";") {
            let raw = String(part).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !raw.isEmpty, let separator = raw.lastIndex(of: "=") else {
                continue
            }
            let path = String(raw[..<separator]).trimmingCharacters(in: .whitespacesAndNewlines)
            let rankString = String(raw[raw.index(after: separator)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty, let rank = Int(rankString), rank > 0 else {
                continue
            }
            targets.append(MFluxLoRATargetPayload(modulePath: path, rank: rank))
        }

        return targets
    }

    private static func writeJSON<T: Encodable>(_ object: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(object)
        try data.write(to: url, options: [.atomic])
    }

    private static func normalizedValue(_ rawValue: String?) -> String? {
        guard let rawValue else { return nil }
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func intValue(_ rawValue: String?) -> Int? {
        guard let normalized = normalizedValue(rawValue) else { return nil }
        return Int(normalized)
    }

    private static func doubleValue(_ rawValue: String?) -> Double? {
        guard let normalized = normalizedValue(rawValue) else { return nil }
        return Double(normalized)
    }
}

struct MFluxLossEntry: Codable, Sendable, Hashable {
    let step: Int
    let loss: Double

    init(step: Int, loss: Double) {
        self.step = step
        self.loss = loss
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.step = try container.decode(LenientInt.self, forKey: .step).value
        self.loss = try container.decode(LenientDouble.self, forKey: .loss).value
    }
}

struct MFluxCheckpointManifest: Codable, Sendable, Hashable {
    let metadata: Metadata?
    let files: [String: String]

    struct Metadata: Codable, Sendable, Hashable {
        let numberOfTrainingData: Int

        private enum CodingKeys: String, CodingKey {
            case numberOfTrainingData = "number_of_training_data"
        }
    }
}

private struct MFluxConfigPayload: Encodable, Sendable, Hashable {
    let model: String
    let seed: UInt64
    let steps: Int
    let data: String
    let trainingLoop: MFluxTrainingLoopPayload
    let optimizer: MFluxOptimizerPayload
    let checkpoint: MFluxCheckpointPayload
    let maxResolution: Int?
    let loraLayers: MFluxLoRALayersPayload?

    private enum CodingKeys: String, CodingKey {
        case model
        case seed
        case steps
        case data
        case trainingLoop = "training_loop"
        case optimizer
        case checkpoint
        case maxResolution = "max_resolution"
        case loraLayers = "lora_layers"
    }
}

private struct MFluxTrainingLoopPayload: Encodable, Sendable, Hashable {
    let batchSize: Int
    let numEpochs: Int
    let timestepLow: Int?
    let timestepHigh: Int?

    private enum CodingKeys: String, CodingKey {
        case batchSize = "batch_size"
        case numEpochs = "num_epochs"
        case timestepLow = "timestep_low"
        case timestepHigh = "timestep_high"
    }
}

private struct MFluxOptimizerPayload: Encodable, Sendable, Hashable {
    let name: String
    let learningRate: Double

    private enum CodingKeys: String, CodingKey {
        case name
        case learningRate = "learning_rate"
    }
}

private struct MFluxCheckpointPayload: Encodable, Sendable, Hashable {
    let saveFrequency: Int
    let outputPath: String

    private enum CodingKeys: String, CodingKey {
        case saveFrequency = "save_frequency"
        case outputPath = "output_path"
    }
}

private struct MFluxLoRALayersPayload: Encodable, Sendable, Hashable {
    let targets: [MFluxLoRATargetPayload]
}

struct MFluxLoRATargetPayload: Codable, Sendable, Hashable {
    let modulePath: String
    let rank: Int

    private enum CodingKeys: String, CodingKey {
        case modulePath = "module_path"
        case rank
    }
}
