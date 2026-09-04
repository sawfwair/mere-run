import Foundation
import MLX

#if canImport(CoreML)
import CoreML

final class ParakeetCoreMLDecoder: ParakeetExternalTDTDecoder {
    private let manifest: ParakeetCoreMLManifest.CoreMLDecoder
    private let model: MLModel
    private let embeddingData: Data
    private let vocabulary: [String]
    private let durations: [Int]
    private let maxSymbols: Int?
    private let stepSeconds: TimeInterval

    var maximumBatchSize: Int { manifest.lanes }

    init(
        artifactURL: URL,
        manifest: ParakeetCoreMLManifest,
        config: ParakeetModelConfig
    ) throws {
        guard let decoder = manifest.coreMLDecoder else {
            throw ParakeetCoreMLError.incompatibleManifest
        }
        let decoderURL = artifactURL.appendingPathComponent(
            decoder.compiledModelDirectory,
            isDirectory: true
        )
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: decoderURL, configuration: configuration)

        let embeddingURL = artifactURL.appendingPathComponent(decoder.embeddingFile)
        let embedding = try Data(contentsOf: embeddingURL, options: .mappedIfSafe)
        let expectedEmbeddingBytes = (decoder.vocabularySize + 1) * decoder.hiddenSize * 2
        guard embedding.count == expectedEmbeddingBytes else {
            throw ParakeetCoreMLError.invalidEmbeddingByteCount(
                expected: expectedEmbeddingBytes,
                actual: embedding.count
            )
        }

        self.manifest = decoder
        self.embeddingData = embedding
        self.vocabulary = config.vocabulary
        self.durations = config.tdtDurations ?? [0, 1, 2, 3, 4]
        self.maxSymbols = config.maxSymbols
        self.stepSeconds = TimeInterval(
            config.encoder.subsamplingFactor * config.preprocessor.hopLength
        ) / TimeInterval(config.preprocessor.sampleRate)
    }

    func decode(
        encoded: MLXArray,
        lengths: [Int]
    ) throws -> [ParakeetAlignedResult] {
        let batch = encoded.dim(0)
        guard batch <= manifest.lanes,
              encoded.ndim == 3,
              encoded.dim(2) == 1_024,
              lengths.count == batch else {
            throw ParakeetCoreMLError.unsupportedDecoderInputShape(encoded.shape)
        }

        let encodedValues = encoded.asType(.float32).asArray(Float.self).map(Float16.init)
        let encoderWindow = try MLMultiArray(
            shape: [
                NSNumber(value: manifest.lanes),
                NSNumber(value: manifest.windowFrames),
                1_024,
            ],
            dataType: .float16
        )
        let tokenEmbedding = try MLMultiArray(
            shape: [NSNumber(value: manifest.lanes), NSNumber(value: manifest.hiddenSize)],
            dataType: .float16
        )
        zero(encoderWindow)
        zero(tokenEmbedding)

        let hiddenState = try stateArray()
        let cellState = try stateArray()
        let nextHidden = try stateArray()
        let nextCell = try stateArray()
        let tokenOutput = try MLMultiArray(
            shape: [NSNumber(value: manifest.lanes), NSNumber(value: manifest.windowFrames)],
            dataType: .int32
        )
        let durationOutput = try MLMultiArray(
            shape: [NSNumber(value: manifest.lanes), NSNumber(value: manifest.windowFrames)],
            dataType: .int32
        )

        let inputs = DecoderFeatureProvider(
            manifest: manifest,
            encoderWindow: encoderWindow,
            tokenEmbedding: tokenEmbedding,
            hiddenState: hiddenState,
            cellState: cellState
        )
        let predictionOptions = MLPredictionOptions()

        var cursors = [Int](repeating: 0, count: batch)
        var lastTokens = [Int](repeating: vocabulary.count, count: batch)
        var newSymbols = [Int](repeating: 0, count: batch)
        var alignedTokens = [[ParakeetAlignedToken]](repeating: [], count: batch)

        while (0..<batch).contains(where: { cursors[$0] < min(lengths[$0], encoded.dim(1)) }) {
            fillEncoderWindow(
                from: encodedValues,
                encodedShape: encoded.shape,
                lengths: lengths,
                cursors: cursors,
                into: encoderWindow
            )
            fillTokenEmbeddings(lastTokens, into: tokenEmbedding)
            predictionOptions.outputBackings = [
                manifest.tokenOutputName: tokenOutput,
                manifest.durationOutputName: durationOutput,
                manifest.hiddenOutputName: nextHidden,
                manifest.cellOutputName: nextCell,
            ]

            let prediction = try model.prediction(from: inputs, options: predictionOptions)
            guard prediction.featureValue(for: manifest.tokenOutputName)?.multiArrayValue != nil,
                  prediction.featureValue(for: manifest.durationOutputName)?.multiArrayValue != nil,
                  prediction.featureValue(for: manifest.hiddenOutputName)?.multiArrayValue != nil,
                  prediction.featureValue(for: manifest.cellOutputName)?.multiArrayValue != nil else {
                throw ParakeetCoreMLError.missingOutput("Core ML decoder output")
            }

            for lane in 0..<batch {
                let maxLength = min(lengths[lane], encoded.dim(1))
                guard cursors[lane] < maxLength else { continue }
                let windowStart = cursors[lane]
                let windowEnd = min(windowStart + manifest.windowFrames, maxLength)
                var emitted = false

                while cursors[lane] < windowEnd {
                    let row = cursors[lane] - windowStart
                    let outputIndex = lane * manifest.windowFrames + row
                    let predictedToken = Int(
                        tokenOutput.dataPointer.assumingMemoryBound(to: Int32.self)[outputIndex]
                    )
                    let decisionIndex = Int(
                        durationOutput.dataPointer.assumingMemoryBound(to: Int32.self)[outputIndex]
                    )
                    let clampedDecision = min(max(0, decisionIndex), max(0, durations.count - 1))
                    let durationSteps = max(0, durations[clampedDecision])

                    if predictedToken != vocabulary.count {
                        alignedTokens[lane].append(
                            ParakeetAlignedToken(
                                id: predictedToken,
                                text: ParakeetTokenizer.decode(
                                    tokens: [predictedToken],
                                    vocabulary: vocabulary
                                ),
                                start: TimeInterval(cursors[lane]) * stepSeconds,
                                duration: TimeInterval(durationSteps) * stepSeconds
                            )
                        )
                        lastTokens[lane] = predictedToken
                        copyStateLane(
                            lane,
                            from: nextHidden,
                            to: hiddenState
                        )
                        copyStateLane(
                            lane,
                            from: nextCell,
                            to: cellState
                        )
                        emitted = true
                    }

                    cursors[lane] += durationSteps
                    newSymbols[lane] += 1
                    if durationSteps != 0 {
                        newSymbols[lane] = 0
                    } else if let maxSymbols, maxSymbols <= newSymbols[lane] {
                        cursors[lane] += 1
                        newSymbols[lane] = 0
                    }

                    if emitted {
                        break
                    }
                    if durationSteps == 0, maxSymbols == nil {
                        break
                    }
                }
            }

        }

        return alignedTokens.map { tokens in
            ParakeetAlignment.sentencesToResult(
                ParakeetAlignment.tokensToSentences(tokens)
            )
        }
    }

    private func stateArray() throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [
                NSNumber(value: manifest.layers),
                NSNumber(value: manifest.lanes),
                NSNumber(value: manifest.hiddenSize),
            ],
            dataType: .float16
        )
        zero(array)
        return array
    }

    private func fillEncoderWindow(
        from source: [Float16],
        encodedShape: [Int],
        lengths: [Int],
        cursors: [Int],
        into destination: MLMultiArray
    ) {
        zero(destination)
        let features = encodedShape[2]
        let availableFrames = encodedShape[1]
        let pointer = destination.dataPointer.assumingMemoryBound(to: UInt16.self)
        for lane in 0..<min(encodedShape[0], manifest.lanes) {
            let frameCount = min(
                manifest.windowFrames,
                min(lengths[lane], availableFrames) - cursors[lane]
            )
            guard frameCount > 0 else { continue }
            let sourceOffset = (lane * availableFrames + cursors[lane]) * features
            let destinationOffset = lane * manifest.windowFrames * features
            for index in 0..<(frameCount * features) {
                pointer[destinationOffset + index] = source[sourceOffset + index].bitPattern
            }
        }
    }

    private func fillTokenEmbeddings(
        _ tokens: [Int],
        into destination: MLMultiArray
    ) {
        zero(destination)
        let bytesPerEmbedding = manifest.hiddenSize * 2
        embeddingData.withUnsafeBytes { rawBuffer in
            guard let sourceBase = rawBuffer.baseAddress else { return }
            for lane in tokens.indices where tokens[lane] != vocabulary.count {
                let source = sourceBase.advanced(by: tokens[lane] * bytesPerEmbedding)
                let destinationPointer = destination.dataPointer.advanced(
                    by: lane * bytesPerEmbedding
                )
                destinationPointer.copyMemory(from: source, byteCount: bytesPerEmbedding)
            }
        }
    }

    private func copyStateLane(
        _ lane: Int,
        from source: MLMultiArray,
        to destination: MLMultiArray
    ) {
        let sourcePointer = source.dataPointer.assumingMemoryBound(to: UInt16.self)
        let destinationPointer = destination.dataPointer.assumingMemoryBound(to: UInt16.self)
        for layer in 0..<manifest.layers {
            let offset = (layer * manifest.lanes + lane) * manifest.hiddenSize
            destinationPointer.advanced(by: offset).update(
                from: sourcePointer.advanced(by: offset),
                count: manifest.hiddenSize
            )
        }
    }

    private func zero(_ array: MLMultiArray) {
        array.dataPointer.initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: array.count * Self.byteCount(for: array.dataType)
        )
    }

    private static func byteCount(for type: MLMultiArrayDataType) -> Int {
        switch type {
        case .int8:
            1
        case .float16:
            2
        case .float32, .int32:
            4
        case .double:
            8
        @unknown default:
            8
        }
    }
}

private final class DecoderFeatureProvider: NSObject, MLFeatureProvider {
    let manifest: ParakeetCoreMLManifest.CoreMLDecoder
    let encoderWindow: MLMultiArray
    let tokenEmbedding: MLMultiArray
    let hiddenState: MLMultiArray
    let cellState: MLMultiArray

    init(
        manifest: ParakeetCoreMLManifest.CoreMLDecoder,
        encoderWindow: MLMultiArray,
        tokenEmbedding: MLMultiArray,
        hiddenState: MLMultiArray,
        cellState: MLMultiArray
    ) {
        self.manifest = manifest
        self.encoderWindow = encoderWindow
        self.tokenEmbedding = tokenEmbedding
        self.hiddenState = hiddenState
        self.cellState = cellState
    }

    var featureNames: Set<String> {
        [
            manifest.encoderInputName,
            manifest.embeddingInputName,
            manifest.hiddenInputName,
            manifest.cellInputName,
        ]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case manifest.encoderInputName:
            MLFeatureValue(multiArray: encoderWindow)
        case manifest.embeddingInputName:
            MLFeatureValue(multiArray: tokenEmbedding)
        case manifest.hiddenInputName:
            MLFeatureValue(multiArray: hiddenState)
        case manifest.cellInputName:
            MLFeatureValue(multiArray: cellState)
        default:
            nil
        }
    }
}
#else
final class ParakeetCoreMLDecoder: ParakeetExternalTDTDecoder {
    var maximumBatchSize: Int { 1 }

    init(
        artifactURL: URL,
        manifest: ParakeetCoreMLManifest,
        config: ParakeetModelConfig
    ) throws {
        throw ParakeetError.coreMLUnavailable
    }

    func decode(
        encoded: MLXArray,
        lengths: [Int]
    ) throws -> [ParakeetAlignedResult] {
        throw ParakeetError.coreMLUnavailable
    }
}
#endif
