import AudioSortformer
import Foundation
import MereRunCore
import MLX
import MLXNN

public enum Nemotron3DiarizationLoadError: LocalizedError {
    case archiveExtractionFailed
    case unexpectedTensorCount(Int)

    public var errorDescription: String? {
        switch self {
        case .archiveExtractionFailed:
            "The pinned Nemotron 3 NeMo archive could not be unpacked."
        case .unexpectedTensorCount(let count):
            "Nemotron 3 checkpoint has \(count) tensors; expected 363."
        }
    }
}

/// Loads the pinned NeMo initializer with Mere's non-executing PyTorch state-dict
/// reader, then runs the model through native Swift/MLX computation.
public final class Nemotron3Diarizer {
    private let runtime: Nemotron3DiarizationRuntime

    public init(modelDirectory: URL) throws {
        let archiveURL = try Nemotron3DiarizationResources.verify(at: modelDirectory)
        let extractionRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-nemotron3-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: extractionRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: extractionRoot) }
        let checkpointURL = extractionRoot.appendingPathComponent("model_weights.ckpt")
        FileManager.default.createFile(atPath: checkpointURL.path, contents: nil)
        let checkpointHandle = try FileHandle(forWritingTo: checkpointURL)
        let extraction = Process()
        extraction.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        extraction.arguments = ["-xOf", archiveURL.path, "model_weights.ckpt"]
        extraction.standardOutput = checkpointHandle
        extraction.standardError = Pipe()
        try extraction.run()
        extraction.waitUntilExit()
        try checkpointHandle.close()
        guard extraction.terminationStatus == 0,
              try ModelArtifactPin.fileSHA256(checkpointURL)
                  == Nemotron3DiarizationResources.checkpointSHA256 else {
            throw Nemotron3DiarizationLoadError.archiveExtractionFailed
        }

        let source = try PyTorchStateDictArchive(url: checkpointURL, verifyEntryChecksums: false)
        guard source.tensors.count == 363 else {
            throw Nemotron3DiarizationLoadError.unexpectedTensorCount(source.tensors.count)
        }
        let model = Nemotron3DiarizationModel()
        var weights: [String: MLXArray] = [:]
        for descriptor in source.tensors {
            let name = descriptor.name
            guard name.hasPrefix("encoder.") || name.hasPrefix("sortformer_modules.") else { continue }
            guard !name.hasPrefix("sortformer_modules.activity_head.")
                    && !name.hasPrefix("sortformer_modules.hidden_to_spks.") else { continue }
            weights[name] = try source.loadArray(for: descriptor, dtype: .bfloat16)
        }
        try model.update(
            parameters: ModuleParameters.unflattened(Nemotron3DiarizationModel.compatibleWeights(weights)),
            verify: .all
        )
        eval(model.parameters())

        let window = try source.loadArray(named: "preprocessor.featurizer.window", dtype: .float32)
        let filterbank = try source.loadArray(named: "preprocessor.featurizer.fb", dtype: .float32)
        eval(window, filterbank)
        runtime = Nemotron3DiarizationRuntime(model: model, window: window, filterbank: filterbank)
    }

    public func diarize(
        samples: [Float],
        sampleRate: Int = 16_000,
        threshold: Float = 0.5,
        minDuration: Float = 0.25,
        mergeGap: Float = 0.25,
        chunkLength: Int = 340,
        rightContext: Int = 40,
        fifoLength: Int = 40,
        cacheUpdatePeriod: Int = 300
    ) throws -> DiarizationOutput {
        try runtime.diarize(
            samples: samples,
            sampleRate: sampleRate,
            threshold: threshold,
            minDuration: minDuration,
            mergeGap: mergeGap,
            chunkLength: chunkLength,
            rightContext: rightContext,
            fifoLength: fifoLength,
            cacheUpdatePeriod: cacheUpdatePeriod
        )
    }
}
