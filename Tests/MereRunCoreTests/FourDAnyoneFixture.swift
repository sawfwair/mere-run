import Foundation
import MLX
import XCTest
@testable import MereRunCore

enum FourDAnyoneFixture {
    static let configuration = Wan2TransformerConfiguration(
        inputChannels: 4, hiddenSize: 24, feedForwardSize: 40,
        timestepFrequencySize: 8, textEmbeddingSize: 16,
        outputChannels: 4, headCount: 2, layerCount: 2
    )

    static func url(_ name: String, extension suffix: String) -> URL {
        Bundle.module.url(forResource: name, withExtension: suffix, subdirectory: "Fixtures/FourDAnyone")!
    }

    static func tensors() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url("reference", extension: "safetensors"))
    }

    static func model() throws -> FourDAnyoneTransformerModel {
        try FourDAnyoneModelLoader.loadTransformer(
            from: url("transformer", extension: "safetensors"), configuration: configuration, dtype: .float32
        )
    }

    static func input(_ tensors: [String: MLXArray], packed: Bool) -> FourDAnyoneTransformerInput {
        FourDAnyoneTransformerInput(
            latents: tensors["latents"]!, sources: tensors["sources"]![0..<(packed ? 5 : 1)],
            poseFeatures: tensors["poses"]!, nullPoseFeatures: tensors["null"]![0..<(packed ? 2 : 1)],
            promptContext: tensors["context"]!, timestep: 625
        )
    }

    static func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 3e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        guard actual.shape == expected.shape else { return }
        let error = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
        XCTAssertTrue(error.isFinite, "Non-finite output", file: file, line: line)
        XCTAssertLessThanOrEqual(error, tolerance, "Maximum absolute error: \(error)", file: file, line: line)
    }

    static func manifest() throws -> Manifest {
        try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url("manifest", extension: "json")))
    }

    struct Manifest: Decodable {
        let upstreamRevision: String
        let modelRevision: String
        let plans: [Plan]
        let poseWeights: [PoseWeight]
    }

    struct Plan: Decodable {
        let viewsPerLayer: Int
        let pitches: [Int]
        let routing: Bool
        let order: [Int]
        let routes: [[[Int]]]
    }

    struct PoseWeight: Decodable {
        let key: String
        let shape: [Int]
        let index: Int
        let amplitude: Float
        let offset: Float
    }
}
