import Foundation
import MLX
import MereRunCore

#if canImport(CoreML)
import CoreML
#endif

struct ParakeetCoreMLManifest: Decodable, Sendable {
    static let filename = "parakeet-coreml.json"
    static let supportedSchemaVersions = Set([1, 2, 3, 4])
    static let sourceRepository = "nvidia/parakeet-tdt-0.6b-v3"
    static let sourceRevision = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    static let sourceLicense = "CC-BY-4.0"
    static let converter = "convert_parakeet_coreml.py"
    static let currentConverterVersion = 4
    static let torchVersion = "2.7.0"
    static let transformersVersion = "5.16.1"
    static let coreMLToolsVersion = "9.0"
    static let xcodeVersion = "Xcode 26.4 / Build version 17E192"
    static let compiledModelDirectory = "encoder.mlmodelc"
    static let compiledDecoderModelDirectory = "decoder.mlmodelc"

    struct Source: Decodable, Sendable {
        let repository: String
        let revision: String
        let license: String
    }

    struct Conversion: Decodable, Sendable {
        let converter: String
        let converterVersion: Int
        let python: String
        let torch: String
        let transformers: String
        let coremltools: String
        let xcode: String
    }

    struct Encoder: Decodable, Sendable {
        let compiledModelDirectory: String
        let inputName: String
        let attentionMaskInputName: String
        let outputName: String
        let outputMaskName: String
        let inputFrames: Int
        let inputFeatures: Int
        let outputFeatures: Int
        let sampleRate: Int
        let windowSeconds: Double
    }

    struct Decoder: Decodable, Sendable {
        let format: String
        let weightsFile: String
        let configFile: String
        let vocabularyFile: String
        let tensorCount: Int
    }

    struct CoreMLDecoder: Decodable, Sendable {
        enum DecisionEncoding: String, Decodable, Sendable {
            case base128Float16 = "base128-float16"
        }

        let compiledModelDirectory: String
        let embeddingFile: String
        let encoderInputName: String
        let embeddingInputName: String
        let hiddenInputName: String
        let cellInputName: String
        let tokenOutputName: String
        let durationOutputName: String
        let hiddenOutputName: String
        let cellOutputName: String
        let lanes: Int
        let windowFrames: Int
        let hiddenSize: Int
        let layers: Int
        let vocabularySize: Int
        var decisionEncoding: DecisionEncoding?

        var usesANESelection: Bool { decisionEncoding == .base128Float16 }
    }

    let schemaVersion: Int
    let source: Source
    let conversion: Conversion
    let encoder: Encoder
    let decoder: Decoder?
    let coreMLDecoder: CoreMLDecoder?
    let artifacts: [ModelArtifactPin]

    static func load(
        artifactURL: URL,
        config: ParakeetModelConfig,
        fileManager: FileManager = .default
    ) throws -> (manifest: ParakeetCoreMLManifest, modelURL: URL) {
        let root = artifactURL.standardizedFileURL
        let manifestURL = root.appendingPathComponent(filename)
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            throw ParakeetCoreMLError.missingManifest(manifestURL.path)
        }

        let manifest = try JSONDecoder().decode(
            ParakeetCoreMLManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        try manifest.validate(config: config)

        for artifact in manifest.artifacts {
            try validateRelativePath(artifact.filename, under: root)
            guard manifest.schemaVersion >= 2 || artifact.filename.hasPrefix(
                manifest.encoder.compiledModelDirectory + "/"
            ) else {
                throw ParakeetCoreMLError.unsafeArtifactPath(artifact.filename)
            }
        }

        try validateRelativePath(manifest.encoder.compiledModelDirectory, under: root)
        let modelURL = root.appendingPathComponent(
            manifest.encoder.compiledModelDirectory,
            isDirectory: true
        )
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw ParakeetCoreMLError.missingCompiledModel(modelURL.path)
        }
        if let decoder = manifest.coreMLDecoder {
            try validateRelativePath(decoder.compiledModelDirectory, under: root)
            try validateRelativePath(decoder.embeddingFile, under: root)
            let decoderURL = root.appendingPathComponent(
                decoder.compiledModelDirectory,
                isDirectory: true
            )
            isDirectory = false
            guard fileManager.fileExists(atPath: decoderURL.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw ParakeetCoreMLError.missingCompiledDecoder(decoderURL.path)
            }
        }

        let expectedFiles = manifest.artifacts.map(\.filename).sorted()
        let actualFiles: [String]
        if manifest.schemaVersion >= 2 {
            actualFiles = try artifactFiles(
                in: root,
                relativeTo: root,
                fileManager: fileManager
            )
            .filter { $0 != Self.filename }
            .sorted()
        } else {
            actualFiles = try artifactFiles(
                in: modelURL,
                relativeTo: root,
                fileManager: fileManager
            ).sorted()
        }
        guard expectedFiles == actualFiles else {
            throw ParakeetCoreMLError.artifactClosureMismatch(
                expected: expectedFiles,
                actual: actualFiles
            )
        }
        for artifact in manifest.artifacts {
            _ = try artifact.verify(in: root, fileManager: fileManager)
        }
        return (manifest, modelURL)
    }

    private func validate(config: ParakeetModelConfig) throws {
        guard Self.supportedSchemaVersions.contains(schemaVersion) else {
            throw ParakeetCoreMLError.unsupportedSchemaVersion(schemaVersion)
        }
        guard source.repository == Self.sourceRepository,
              source.revision == Self.sourceRevision,
              source.license.uppercased() == Self.sourceLicense else {
            throw ParakeetCoreMLError.untrustedSource(
                repository: source.repository,
                revision: source.revision,
                license: source.license
            )
        }
        let expectedConverterVersion: Int
        switch schemaVersion {
        case 1:
            expectedConverterVersion = 1
        case 2:
            expectedConverterVersion = 2
        case 3:
            expectedConverterVersion = 3
        default:
            expectedConverterVersion = Self.currentConverterVersion
        }
        guard conversion.converter == Self.converter,
              conversion.converterVersion == expectedConverterVersion,
              conversion.python.hasPrefix("3.12."),
              conversion.torch == Self.torchVersion,
              conversion.transformers == Self.transformersVersion,
              conversion.coremltools == Self.coreMLToolsVersion,
              conversion.xcode == Self.xcodeVersion else {
            throw ParakeetCoreMLError.untrustedConversion
        }
        guard encoder.compiledModelDirectory == Self.compiledModelDirectory,
              encoder.inputName == "input_features",
              encoder.attentionMaskInputName == "attention_mask",
              encoder.outputName == "encoded_features",
              encoder.outputMaskName == "encoded_attention_mask",
              encoder.inputFrames == 1_501,
              encoder.inputFeatures == config.preprocessor.features,
              encoder.outputFeatures == config.encoder.modelDim,
              encoder.sampleRate == config.preprocessor.sampleRate,
              encoder.windowSeconds == 15 else {
            throw ParakeetCoreMLError.incompatibleManifest
        }
        guard !artifacts.isEmpty else {
            throw ParakeetCoreMLError.emptyArtifactClosure
        }
        let artifactNames = artifacts.map(\.filename)
        guard Set(artifactNames).count == artifactNames.count else {
            throw ParakeetCoreMLError.incompatibleManifest
        }
        if schemaVersion >= 2 {
            guard let decoder,
                  config.packaging == .coreMLHybrid,
                  decoder.format == ParakeetPackaging.coreMLHybrid.rawValue,
                  decoder.weightsFile == "model.safetensors",
                  decoder.configFile == "config.json",
                  decoder.vocabularyFile == "vocab.txt",
                  decoder.tensorCount == 13,
                  artifactNames.contains(decoder.weightsFile),
                  artifactNames.contains(decoder.configFile),
                  artifactNames.contains(decoder.vocabularyFile) else {
                throw ParakeetCoreMLError.incompatibleManifest
            }
        } else if decoder != nil {
            throw ParakeetCoreMLError.incompatibleManifest
        }
        if schemaVersion >= 3 {
            guard let coreMLDecoder,
                  coreMLDecoder.compiledModelDirectory == Self.compiledDecoderModelDirectory,
                  coreMLDecoder.embeddingFile == "embedding.f16",
                  coreMLDecoder.encoderInputName == "encoder_window",
                  coreMLDecoder.embeddingInputName == "token_embedding",
                  coreMLDecoder.hiddenInputName == "hidden_state",
                  coreMLDecoder.cellInputName == "cell_state",
                  coreMLDecoder.tokenOutputName == "token",
                  coreMLDecoder.durationOutputName == "duration",
                  coreMLDecoder.hiddenOutputName == "next_hidden",
                  coreMLDecoder.cellOutputName == "next_cell",
                  coreMLDecoder.lanes == 16,
                  coreMLDecoder.windowFrames == 8,
                  coreMLDecoder.hiddenSize == 640,
                  coreMLDecoder.layers == 2,
                  coreMLDecoder.vocabularySize == config.vocabulary.count,
                  coreMLDecoder.usesANESelection == (schemaVersion == 4),
                  artifactNames.contains(coreMLDecoder.embeddingFile),
                  artifactNames.contains(where: {
                      $0.hasPrefix(coreMLDecoder.compiledModelDirectory + "/")
                  }) else {
                throw ParakeetCoreMLError.incompatibleManifest
            }
        } else if coreMLDecoder != nil {
            throw ParakeetCoreMLError.incompatibleManifest
        }
    }

    private static func validateRelativePath(_ path: String, under root: URL) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.contains("\\"),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw ParakeetCoreMLError.unsafeArtifactPath(path)
        }
        let rootPath = root.standardizedFileURL.path
        let candidatePath = root.appendingPathComponent(path).standardizedFileURL.path
        guard candidatePath == rootPath || candidatePath.hasPrefix(rootPath + "/") else {
            throw ParakeetCoreMLError.unsafeArtifactPath(path)
        }
    }

    private static func artifactFiles(
        in directory: URL,
        relativeTo root: URL,
        fileManager: FileManager
    ) throws -> [String] {
        let keys: Set<URLResourceKey> = [
            .isDirectoryKey,
            .isRegularFileKey,
            .isSymbolicLinkKey,
        ]
        var files: [String] = []
        for child in try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: Array(keys),
            options: []
        ) {
            let values = try child.resourceValues(forKeys: keys)
            if values.isSymbolicLink == true {
                throw ParakeetCoreMLError.symlinkedArtifact(child.path)
            }
            if values.isDirectory == true {
                files += try artifactFiles(
                    in: child,
                    relativeTo: root,
                    fileManager: fileManager
                )
                continue
            }
            guard values.isRegularFile == true else {
                throw ParakeetCoreMLError.unsupportedArtifact(child.path)
            }
            let rootPath = root.standardizedFileURL.path + "/"
            let childPath = child.standardizedFileURL.path
            guard childPath.hasPrefix(rootPath) else {
                throw ParakeetCoreMLError.unsafeArtifactPath(child.path)
            }
            files.append(String(childPath.dropFirst(rootPath.count)))
        }
        return files
    }
}

enum ParakeetCoreMLError: LocalizedError {
    case missingManifest(String)
    case missingCompiledModel(String)
    case missingCompiledDecoder(String)
    case unsupportedSchemaVersion(Int)
    case untrustedSource(repository: String, revision: String, license: String)
    case untrustedConversion
    case incompatibleManifest
    case emptyArtifactClosure
    case unsafeArtifactPath(String)
    case artifactClosureMismatch(expected: [String], actual: [String])
    case symlinkedArtifact(String)
    case unsupportedArtifact(String)
    case unsupportedInputShape([Int])
    case unsupportedDecoderInputShape([Int])
    case inputTooLong(actual: Int, maximum: Int)
    case invalidEmbeddingByteCount(expected: Int, actual: Int)
    case missingOutput(String)
    case unsupportedMultiArrayType(String)
    case invalidOutputShape([Int])

    var errorDescription: String? {
        switch self {
        case .missingManifest(let path):
            return "The Parakeet Core ML manifest is missing: \(path)"
        case .missingCompiledModel(let path):
            return "The compiled Parakeet Core ML encoder is missing: \(path)"
        case .missingCompiledDecoder(let path):
            return "The compiled Parakeet Core ML decoder is missing: \(path)"
        case .unsupportedSchemaVersion(let version):
            return "Unsupported Parakeet Core ML manifest schema version: \(version)."
        case .untrustedSource(let repository, let revision, let license):
            return "Untrusted Parakeet Core ML source: \(repository)@\(revision) (\(license))."
        case .untrustedConversion:
            return "The Parakeet Core ML artifact was not built with Mere's pinned conversion environment."
        case .incompatibleManifest:
            return "The Parakeet Core ML artifact is incompatible with the selected decoder checkpoint."
        case .emptyArtifactClosure:
            return "The Parakeet Core ML manifest does not pin any compiled artifacts."
        case .unsafeArtifactPath(let path):
            return "Unsafe path in the Parakeet Core ML manifest: \(path)"
        case .artifactClosureMismatch(let expected, let actual):
            return "The Parakeet Core ML artifact closure differs from its manifest: expected \(expected), found \(actual)."
        case .symlinkedArtifact(let path):
            return "The Parakeet Core ML artifact contains a symbolic link: \(path)"
        case .unsupportedArtifact(let path):
            return "The Parakeet Core ML artifact contains an unsupported file type: \(path)"
        case .unsupportedInputShape(let shape):
            return "The Parakeet Core ML encoder requires [1, frames, features]; found \(shape)."
        case .unsupportedDecoderInputShape(let shape):
            return "The Parakeet Core ML decoder received an unsupported encoder shape: \(shape)."
        case .inputTooLong(let actual, let maximum):
            return "The Parakeet Core ML encoder accepts at most \(maximum) mel frames; found \(actual)."
        case .invalidEmbeddingByteCount(let expected, let actual):
            return "The Parakeet Core ML decoder embedding table requires \(expected) bytes; found \(actual)."
        case .missingOutput(let name):
            return "The Parakeet Core ML encoder did not return \(name)."
        case .unsupportedMultiArrayType(let type):
            return "Unsupported Core ML tensor type: \(type)."
        case .invalidOutputShape(let shape):
            return "The Parakeet Core ML encoder returned an invalid shape: \(shape)."
        }
    }
}

#if canImport(CoreML)
final class ParakeetCoreMLEncoder: ParakeetExternalEncoder {
    private let manifest: ParakeetCoreMLManifest
    private let model: MLModel

    convenience init(artifactURL: URL, config: ParakeetModelConfig) throws {
        let loaded = try ParakeetCoreMLManifest.load(artifactURL: artifactURL, config: config)
        try self.init(manifest: loaded.manifest, modelURL: loaded.modelURL)
    }

    init(manifest: ParakeetCoreMLManifest, modelURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        self.manifest = manifest
    }

    func encode(_ mel: MLXArray) throws -> ParakeetEncoderOutput {
        let shape = mel.shape
        guard shape.count == 3,
              shape[0] == 1,
              shape[2] == manifest.encoder.inputFeatures else {
            throw ParakeetCoreMLError.unsupportedInputShape(shape)
        }
        let frameCount = shape[1]
        guard frameCount <= manifest.encoder.inputFrames else {
            throw ParakeetCoreMLError.inputTooLong(
                actual: frameCount,
                maximum: manifest.encoder.inputFrames
            )
        }

        let features = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: manifest.encoder.inputFrames),
                NSNumber(value: manifest.encoder.inputFeatures),
            ],
            dataType: .float16
        )
        try Self.writeMel(mel, frameCount: frameCount, to: features)

        let attentionMask = try MLMultiArray(
            shape: [1, NSNumber(value: manifest.encoder.inputFrames)],
            dataType: manifest.schemaVersion >= 4 ? .float16 : .int32
        )
        try Self.writeAttentionMask(frameCount: frameCount, to: attentionMask)

        let inputValues: [String: MLFeatureValue] = [
            manifest.encoder.inputName: MLFeatureValue(multiArray: features),
            manifest.encoder.attentionMaskInputName: MLFeatureValue(multiArray: attentionMask),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: inputValues)
        let prediction = try model.prediction(from: provider)

        guard let encodedArray = prediction.featureValue(
            for: manifest.encoder.outputName
        )?.multiArrayValue else {
            throw ParakeetCoreMLError.missingOutput(manifest.encoder.outputName)
        }
        guard let maskArray = prediction.featureValue(
            for: manifest.encoder.outputMaskName
        )?.multiArrayValue else {
            throw ParakeetCoreMLError.missingOutput(manifest.encoder.outputMaskName)
        }

        let outputShape = encodedArray.shape.map(\.intValue)
        guard outputShape.count == 3,
              outputShape[0] == 1,
              outputShape[2] == manifest.encoder.outputFeatures else {
            throw ParakeetCoreMLError.invalidOutputShape(outputShape)
        }
        let encoded = try Self.readArray(encodedArray)
        let validFrames = max(
            0,
            min(
                try Self.readIntegers(maskArray).reduce(0, +),
                outputShape[1]
            )
        )
        return ParakeetEncoderOutput(features: encoded, lengths: [validFrames])
    }

    private static func writeMel(
        _ mel: MLXArray,
        frameCount: Int,
        to destination: MLMultiArray
    ) throws {
        let values = mel.asType(.float32).asArray(Float.self)
        let expected = frameCount * mel.dim(2)
        guard values.count == expected else {
            throw ParakeetCoreMLError.unsupportedInputShape(mel.shape)
        }
        let offsets = try storageOffsets(destination)
        let pointer = destination.dataPointer.assumingMemoryBound(to: UInt16.self)
        for offset in offsets {
            pointer[offset] = Float16.zero.bitPattern
        }
        for index in values.indices {
            pointer[offsets[index]] = Float16(values[index]).bitPattern
        }
    }

    private static func writeAttentionMask(
        frameCount: Int,
        to destination: MLMultiArray
    ) throws {
        let offsets = try storageOffsets(destination)
        if destination.dataType == .float16 {
            let pointer = destination.dataPointer.assumingMemoryBound(to: UInt16.self)
            for (index, offset) in offsets.enumerated() {
                pointer[offset] = Float16(index < frameCount ? 1 : 0).bitPattern
            }
            return
        }
        let pointer = destination.dataPointer.assumingMemoryBound(to: Int32.self)
        for offset in offsets {
            pointer[offset] = 0
        }
        for index in 0..<frameCount {
            pointer[offsets[index]] = 1
        }
    }

    private static func readArray(_ array: MLMultiArray) throws -> MLXArray {
        let offsets = try storageOffsets(array)
        let shape = array.shape.map(\.intValue)
        switch array.dataType {
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            let values = offsets.map { Float16(bitPattern: pointer[$0]) }
            return MLXArray(values).reshaped(shape)
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            let values = offsets.map { pointer[$0] }
            return MLXArray(values).reshaped(shape)
        case .double:
            let pointer = array.dataPointer.assumingMemoryBound(to: Double.self)
            let values = offsets.map { pointer[$0] }
            return MLXArray(values).reshaped(shape)
        default:
            throw ParakeetCoreMLError.unsupportedMultiArrayType("\(array.dataType)")
        }
    }

    private static func readIntegers(_ array: MLMultiArray) throws -> [Int] {
        let offsets = try storageOffsets(array)
        switch array.dataType {
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            return offsets.map { Int(Float16(bitPattern: pointer[$0])) }
        case .int32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Int32.self)
            return offsets.map { Int(pointer[$0]) }
        default:
            throw ParakeetCoreMLError.unsupportedMultiArrayType("\(array.dataType)")
        }
    }

    private static func storageOffsets(_ array: MLMultiArray) throws -> [Int] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard shape.count == strides.count,
              shape.allSatisfy({ $0 > 0 }),
              strides.allSatisfy({ $0 > 0 }) else {
            throw ParakeetCoreMLError.invalidOutputShape(shape)
        }

        return (0..<array.count).map { linearIndex in
            var remaining = linearIndex
            var storageOffset = 0
            for axis in shape.indices.reversed() {
                let coordinate = remaining % shape[axis]
                remaining /= shape[axis]
                storageOffset += coordinate * strides[axis]
            }
            return storageOffset
        }
    }
}
#else
final class ParakeetCoreMLEncoder: ParakeetExternalEncoder {
    init(artifactURL: URL, config: ParakeetModelConfig) throws {
        throw ParakeetError.coreMLUnavailable
    }

    init(manifest: ParakeetCoreMLManifest, modelURL: URL) throws {
        throw ParakeetError.coreMLUnavailable
    }

    func encode(_ mel: MLXArray) throws -> ParakeetEncoderOutput {
        throw ParakeetError.coreMLUnavailable
    }
}
#endif
