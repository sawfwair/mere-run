#if canImport(CoreML)
import CoreML
import XCTest
@testable import AudioSTT

final class ParakeetCoreMLDecoderOutputTests: XCTestCase {
    func testReadsReturnedOutputsWithNoncontiguousStrides() throws {
        let tokens = try stridedArray(shape: [2, 2], strides: [8, 2], type: .int32)
        let durations = try stridedArray(shape: [2, 2], strides: [8, 2], type: .int32)
        tokens[[1, 1] as [NSNumber]] = 42
        durations[[1, 1] as [NSNumber]] = 3
        let hidden = try stridedArray(shape: [2, 2, 2], strides: [16, 6, 2], type: .float16)
        let cell = try stridedArray(shape: [2, 2, 2], strides: [16, 6, 1], type: .float16)
        hidden[[1, 1, 1] as [NSNumber]] = 7
        cell[[1, 1, 1] as [NSNumber]] = 9
        let prediction = try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: tokens),
            "duration": MLFeatureValue(multiArray: durations),
            "next_hidden": MLFeatureValue(multiArray: hidden),
            "next_cell": MLFeatureValue(multiArray: cell),
        ])

        let output = try ParakeetCoreMLDecoderOutput(prediction: prediction, manifest: manifest)

        XCTAssertEqual(output.token(lane: 1, frame: 1), 42)
        XCTAssertEqual(output.duration(lane: 1, frame: 1), 3)
        XCTAssertTrue(output.hidden === hidden)
        XCTAssertTrue(output.cell === cell)
        let destination = try MLMultiArray(shape: [2, 2, 2], dataType: .float16)
        for index in 0..<destination.count { destination[index] = -1 }
        ParakeetCoreMLDecoderOutput.copyStateLane(1, from: output.hidden, to: destination)
        XCTAssertEqual(destination[[1, 1, 1] as [NSNumber]].floatValue, 7)
        XCTAssertEqual(destination[[1, 0, 1] as [NSNumber]].floatValue, -1)
        ParakeetCoreMLDecoderOutput.copyStateLane(1, from: output.cell, to: destination)
        XCTAssertEqual(destination[[1, 1, 1] as [NSNumber]].floatValue, 9)
    }

    func testRejectsMissingPredictionOutputs() throws {
        let prediction = try MLDictionaryFeatureProvider(dictionary: [:])
        XCTAssertThrowsError(try ParakeetCoreMLDecoderOutput(prediction: prediction, manifest: manifest)) { error in
            guard case ParakeetCoreMLError.missingOutput("token") = error else {
                return XCTFail("Expected missing token output, found \(error)")
            }
        }
    }

    func testReadsExactBase128DecisionsWithNoncontiguousStrides() throws {
        var aneManifest = manifest
        aneManifest.decisionEncoding = .base128Float16
        let tokens = try stridedArray(shape: [2, 2, 2], strides: [20, 6, 2], type: .float16)
        let durations = try stridedArray(shape: [2, 2], strides: [8, 2], type: .float16)
        // These IDs are not all exactly representable as a single Float16.
        let expected = [[2_049, 4_095], [8_191, 8_192]]
        for lane in 0..<2 {
            for frame in 0..<2 {
                let token = expected[lane][frame]
                tokens[[lane, frame, 0].map(NSNumber.init)] = NSNumber(value: token / 128)
                tokens[[lane, frame, 1].map(NSNumber.init)] = NSNumber(value: token % 128)
                durations[[lane, frame].map(NSNumber.init)] = NSNumber(value: lane + frame)
            }
        }
        let state = try MLMultiArray(shape: [2, 2, 2], dataType: .float16)
        let prediction = try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: tokens),
            "duration": MLFeatureValue(multiArray: durations),
            "next_hidden": MLFeatureValue(multiArray: state),
            "next_cell": MLFeatureValue(multiArray: state),
        ])
        let output = try ParakeetCoreMLDecoderOutput(prediction: prediction, manifest: aneManifest)
        for lane in 0..<2 {
            for frame in 0..<2 {
                XCTAssertEqual(output.token(lane: lane, frame: frame), expected[lane][frame])
                XCTAssertEqual(output.duration(lane: lane, frame: frame), lane + frame)
            }
        }
    }

    func testRejectsUnexpectedOutputShape() throws {
        let tokens = try MLMultiArray(shape: [1], dataType: .int32)
        let prediction = try MLDictionaryFeatureProvider(dictionary: ["token": MLFeatureValue(multiArray: tokens)])
        XCTAssertThrowsError(try ParakeetCoreMLDecoderOutput(prediction: prediction, manifest: manifest)) { error in
            guard case ParakeetCoreMLError.invalidOutputShape([1]) = error else {
                return XCTFail("Expected invalid output shape, found \(error)")
            }
        }
    }

    private var manifest: ParakeetCoreMLManifest.CoreMLDecoder {
        .init(
            compiledModelDirectory: "decoder.mlmodelc", embeddingFile: "embedding.f16",
            encoderInputName: "encoder_window", embeddingInputName: "token_embedding",
            hiddenInputName: "hidden_state", cellInputName: "cell_state",
            tokenOutputName: "token", durationOutputName: "duration",
            hiddenOutputName: "next_hidden", cellOutputName: "next_cell",
            lanes: 2, windowFrames: 2, hiddenSize: 2, layers: 2, vocabularySize: 64
        )
    }

    private func stridedArray(
        shape: [Int], strides: [Int], type: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let storageCount = zip(shape, strides).reduce(1) { $0 + ($1.0 - 1) * $1.1 }
        let bytes = storageCount * (type == .float16 ? 2 : 4)
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16)
        pointer.initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
        return try MLMultiArray(
            dataPointer: pointer, shape: shape.map(NSNumber.init), dataType: type,
            strides: strides.map(NSNumber.init), deallocator: { $0.deallocate() }
        )
    }
}
#endif
