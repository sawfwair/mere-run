#if canImport(CoreML)
import CoreML
import os
#endif
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
import Foundation
import MLX
import AudioCore
import MereRunCore

/// Fixed 30-second Whisper encoder plus SOV's consecutive-frame projector.
/// CPU_AND_NE excludes GPU placement; Instruments evidence is still required to
/// establish which operations actually executed on the Neural Engine.
final class SOVCoreMLEncoder {
    struct Manifest: Decodable {
        struct Artifact: Decodable { let path: String; let sha256: String }
        let schemaVersion: Int
        let sourceRepository: String
        let sourceRevision: String
        let inputShape: [Int]
        let outputShape: [Int]
        let files: [Artifact]

        func validate(root: URL) throws {
            guard schemaVersion == 1, sourceRepository == SOVTranscribeResources.repository,
                  sourceRevision == SOVTranscribeResources.revision,
                  inputShape == [1, 128, 1, 3000], outputShape == [1, 375, 2816], !files.isEmpty,
                  Set(files.map(\.path)).count == files.count else {
                throw SpeechTranscriptionIssue("invalid_coreml_artifact", "Incompatible SOV Core ML manifest.")
            }
            let compiled = root.appendingPathComponent("encoder.mlmodelc").resolvingSymlinksInPath()
            guard compiled.path == root.resolvingSymlinksInPath().appendingPathComponent("encoder.mlmodelc").path else {
                throw SpeechTranscriptionIssue("invalid_coreml_artifact", "SOV compiled model must remain inside its artifact directory.")
            }
            let enumerator = FileManager.default.enumerator(at: compiled, includingPropertiesForKeys: [.isRegularFileKey])
            var actual: Set<String> = []
            while let file = enumerator?.nextObject() as? URL {
                if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                    let canonical = file.resolvingSymlinksInPath()
                    actual.insert("encoder.mlmodelc/" + canonical.pathComponents.dropFirst(compiled.pathComponents.count).joined(separator: "/"))
                }
            }
            guard actual == Set(files.map(\.path)) else {
                throw SpeechTranscriptionIssue("invalid_coreml_artifact", "SOV compiled model file closure differs from its manifest.")
            }
            for artifact in files {
                let components = artifact.path.split(separator: "/", omittingEmptySubsequences: false)
                let url = root.appendingPathComponent(artifact.path).resolvingSymlinksInPath()
                guard components.first == "encoder.mlmodelc", components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                      url.path.hasPrefix(compiled.path + "/") else {
                    throw SpeechTranscriptionIssue("invalid_coreml_artifact", "Invalid SOV compiled artifact path.")
                }
                let digest = SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
                guard digest == artifact.sha256 else {
                    throw SpeechTranscriptionIssue("invalid_coreml_artifact", "SOV compiled artifact checksum mismatch: \(artifact.path).")
                }
            }
        }
    }

    #if canImport(CoreML)
    private let signposter = OSSignposter(subsystem: "run.mere.sov", category: "encoder")
    private let model: MLModel

    init(root: URL) throws {
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: root.appendingPathComponent("sov-coreml.json")))
        try manifest.validate(root: root)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: root.appendingPathComponent("encoder.mlmodelc"), configuration: configuration)
        guard model.modelDescription.inputDescriptionsByName["mel"]?.multiArrayConstraint?.shape.map(\.intValue) == manifest.inputShape,
              model.modelDescription.outputDescriptionsByName["audio_embeddings"]?.multiArrayConstraint?.shape.map(\.intValue) == manifest.outputShape else {
            throw SpeechTranscriptionIssue("invalid_coreml_artifact", "SOV compiled model shapes differ from its manifest.")
        }
    }

    func encode(_ mel: MLXArray) throws -> MLXArray {
        guard mel.shape == [1, 128, 3000] else {
            throw SpeechTranscriptionIssue("invalid_audio", "SOV Core ML expects one complete 30-second mel window.")
        }
        let input = try MLMultiArray(shape: [1, 128, 1, 3000], dataType: .float16)
        let values = mel.asType(.float16).asArray(Float16.self)
        let strides = input.strides.map(\.intValue)
        let pointer = input.dataPointer.assumingMemoryBound(to: Float16.self)
        for channel in 0..<128 {
            for frame in 0..<3000 { pointer[channel * strides[1] + frame * strides[3]] = values[channel * 3000 + frame] }
        }
        let interval = signposter.beginInterval("SOVEncoderProjector")
        defer { signposter.endInterval("SOVEncoderProjector", interval) }
        let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["mel": MLFeatureValue(multiArray: input)]))
        guard let output = prediction.featureValue(for: "audio_embeddings")?.multiArrayValue,
              output.shape.map(\.intValue) == [1, 375, 2816], output.dataType == .float16 else {
            throw SpeechTranscriptionIssue("invalid_coreml_artifact", "Invalid SOV Core ML projector output.")
        }
        let outputStrides = output.strides.map(\.intValue)
        let source = output.dataPointer.assumingMemoryBound(to: Float16.self)
        var result = [Float16](repeating: 0, count: 375 * 2816)
        for frame in 0..<375 {
            for channel in 0..<2816 { result[frame * 2816 + channel] = source[frame * outputStrides[1] + channel * outputStrides[2]] }
        }
        return MLXArray(result).reshaped(1, 375, 2816)
    }
    #else
    init(root: URL) throws {
        throw SpeechTranscriptionIssue("unsupported_provider", "Core ML is unavailable on this platform.")
    }

    func encode(_ mel: MLXArray) throws -> MLXArray {
        throw SpeechTranscriptionIssue("unsupported_provider", "Core ML is unavailable on this platform.")
    }
    #endif

}
