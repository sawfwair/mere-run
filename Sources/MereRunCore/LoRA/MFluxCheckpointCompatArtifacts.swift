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

        var iteratorPayload: [String: Any] = [
            "num_iterations": safeStep,
            "seed": seed,
            "batch_size": max(batchSize, 1),
        ]
        if let iteratorCursor {
            iteratorPayload["current_permutation"] = iteratorCursor.permutation
            iteratorPayload["position"] = iteratorCursor.position
            if let pythonRNG = iteratorCursor.pythonRNG {
                iteratorPayload["rng_state"] = pythonRNG.serializedState()
            }
            if let localRNGState = iteratorCursor.localRNGState {
                iteratorPayload["zero_rng_state"] = String(localRNGState)
            }
        }
        try writeJSON(iteratorPayload, to: iteratorURL)

        let sortedLossPoints = lossPoints.sorted { lhs, rhs in lhs.step < rhs.step }
        let lossPayload = sortedLossPoints.map { point in
            [
                "step": point.step,
                "loss": Double(point.loss),
            ] as [String: Any]
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
        let checkpointManifestPayload: [String: Any] = [
            "metadata": [
                "number_of_training_data": max(datasetCount, 0),
            ],
            "files": [
                "config": configURL.lastPathComponent,
                "optimizer": optimizerFile,
                "lora_adapter": loraAdapterFileName,
                "iterator": iteratorURL.lastPathComponent,
                "loss": lossURL.lastPathComponent,
            ],
        ]
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
    ) -> [String: Any] {
        let modelValue = normalizedValue(snapshot["model"]) ?? "unknown-model"
        let dataValue = normalizedValue(snapshot["dataset_root"]) ?? ""
        let schedulerSteps = intValue(snapshot["scheduler_steps"]) ?? 1000
        let learningRate = doubleValue(snapshot["learning_rate"]) ?? 1e-4
        let trainingLoop: [String: Any] = {
            var payload: [String: Any] = [
                "batch_size": max(batchSize, 1),
                "num_epochs": 1,
            ]
            if let timestepLow = intValue(snapshot["timestep_low"]) {
                payload["timestep_low"] = timestepLow
            }
            if let timestepHigh = intValue(snapshot["timestep_high"]) {
                payload["timestep_high"] = timestepHigh
            }
            return payload
        }()
        let checkpointInterval = intValue(snapshot["checkpoint_interval"]) ?? max(step, 1)
        var payload: [String: Any] = [
            "model": modelValue,
            "seed": seed,
            "steps": max(schedulerSteps, 1),
            "data": dataValue,
            "training_loop": trainingLoop,
            "optimizer": [
                "name": "adamw",
                "learning_rate": learningRate,
            ],
            "checkpoint": [
                "save_frequency": max(checkpointInterval, 1),
                "output_path": checkpointDirectory.path,
            ],
        ]

        if let maxResolution = intValue(snapshot["max_resolution"]), maxResolution > 0 {
            payload["max_resolution"] = maxResolution
        }
        let targets = loraTargets(from: snapshot["lora_target_ranks"])
        if !targets.isEmpty {
            payload["lora_layers"] = ["targets": targets]
        }

        return payload
    }

    private static func loraTargets(from serializedRanks: String?) -> [[String: Any]] {
        guard let serializedRanks = normalizedValue(serializedRanks) else {
            return []
        }

        var targets: [[String: Any]] = []
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
            targets.append([
                "module_path": path,
                "rank": rank,
            ])
        }

        return targets
    }

    private static func writeJSON(_ object: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
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
