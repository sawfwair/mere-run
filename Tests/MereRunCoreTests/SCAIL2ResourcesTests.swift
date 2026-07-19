import Foundation
import XCTest
@testable import MereRunCore

final class SCAIL2ResourcesTests: XCTestCase {
    func testDefaultConfigurationMatchesSCAIL214BContract() throws {
        let configuration = SCAIL2Configuration()
        XCTAssertEqual(configuration.validationIssues(), [])
        XCTAssertEqual(configuration.patchSize, [1, 2, 2])
        XCTAssertEqual(configuration.inputChannels, 20)
        XCTAssertEqual(configuration.maskChannels, 28)
        XCTAssertEqual(configuration.hiddenSize, 5_120)
        XCTAssertEqual(configuration.layerCount, 40)
        XCTAssertEqual(configuration.vaeStride, [4, 8, 8])
        XCTAssertEqual(configuration.segmentLength, 81)
        XCTAssertEqual(configuration.segmentOverlap, 5)
    }

    func testResourcesAcceptShardedTransformerAndTextEncoder() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("scail2-resources-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = SCAIL2Resources(rootURL: root)
        let files = [
            resources.configURL,
            resources.clipURL,
            resources.tokenizerURL,
            resources.vaeURL,
            resources.sourceLicenseURL,
            resources.sourceReadmeURL,
            resources.provenanceURL,
        ]
        for file in files {
            try Data().write(to: file)
        }
        let transformerShard = root.appendingPathComponent("model-00001-of-00001.safetensors")
        let textShard = root.appendingPathComponent("t5_encoder-00001-of-00001.safetensors")
        try Data().write(to: transformerShard)
        try Data().write(to: textShard)
        try Data(
            #"{"weight_map":{"blocks.0.modulation":"model-00001-of-00001.safetensors"}}"#.utf8
        ).write(to: resources.transformerIndexURL)
        try Data(
            #"{"weight_map":{"token_embedding.weight":"t5_encoder-00001-of-00001.safetensors"}}"#.utf8
        ).write(to: resources.textEncoderIndexURL)
        XCTAssertEqual(resources.validate(), [])
    }

    func testResourcesRejectIndexWithMissingShard() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("scail2-missing-shard-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = SCAIL2Resources(rootURL: root)
        try Data(
            #"{"weight_map":{"blocks.0.modulation":"model-00001-of-00001.safetensors"}}"#.utf8
        ).write(to: resources.transformerIndexURL)

        XCTAssertTrue(
            resources.validate().contains {
                $0.lastPathComponent == "model-00001-of-00001.safetensors"
            }
        )
    }
}
