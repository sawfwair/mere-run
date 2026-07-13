import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class MMAudioResourcesTests: XCTestCase {
    func testEightSecond44KSequenceContractMatchesUpstream() {
        let config = MMAudioGenerationConfig()

        XCTAssertEqual(config.latentSequenceLength, 345)
        XCTAssertEqual(config.clipSequenceLength, 64)
        XCTAssertEqual(config.syncSequenceLength, 192)
        XCTAssertEqual(
            config.latentSequenceLength
                * MMAudioResources.spectrogramFrameRate
                * MMAudioResources.latentDownsampleRate,
            353_280
        )
    }

    func testFrameSamplingUsesFirstDecodedFrameAtOrAfterEachTargetTimestamp() {
        XCTAssertEqual(
            MMAudioVideoPreprocessor.sampledFrameIndices(
                sourceFrameRate: 30,
                sourceFrameCount: 240,
                targetFrameRate: 8,
                targetFrameCount: 8
            ),
            [0, 4, 8, 12, 15, 19, 23, 27]
        )
        XCTAssertEqual(
            MMAudioVideoPreprocessor.sampledFrameIndices(
                sourceFrameRate: 24,
                sourceFrameCount: 24,
                targetFrameRate: 25,
                targetFrameCount: 6
            ),
            [0, 1, 2, 3, 4, 5]
        )
    }

    func testNearestExactSequenceInterpolationMatchesPyTorchIndexConvention() {
        let source = MLXArray((0..<16).map(Float.init)).reshaped(1, 16, 1)
        let actual = MMAudioTensorOps.nearestInterpolateSequence(source, length: 44)
        XCTAssertEqual(actual.asArray(Float.self).map(Int.init), [
            0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
            6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11,
            11, 12, 12, 12, 13, 13, 14, 14, 14, 15, 15, 15,
        ])
    }

    func testResourceValidationAcceptsOfficialPyTorchVocoderCheckpoint() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mmaudio-resources-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let resources = MMAudioModelResources(rootURL: root)
        try FileManager.default.createDirectory(
            at: resources.clipTokenizerURL,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: resources.bigVGANURL,
            withIntermediateDirectories: true
        )
        for url in [
            resources.networkWeightsURL,
            resources.clipWeightsURL,
            resources.synchformerWeightsURL,
            resources.vaeWeightsURL,
            resources.clipTokenizerURL.appendingPathComponent("tokenizer.json"),
            resources.bigVGANConfigURL,
            resources.bigVGANURL.appendingPathComponent(MMAudioResources.bigVGANPyTorchFilename),
        ] {
            XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
        }

        XCTAssertEqual(resources.validate(), [])
        XCTAssertEqual(
            resources.bigVGANWeightsURL().lastPathComponent,
            MMAudioResources.bigVGANPyTorchFilename
        )
    }

    func testSafetensorsVocoderCheckpointTakesPrecedence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mmaudio-vocoder-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = MMAudioModelResources(rootURL: root)
        try FileManager.default.createDirectory(
            at: resources.bigVGANURL,
            withIntermediateDirectories: true
        )
        for filename in [
            MMAudioResources.bigVGANPyTorchFilename,
            MMAudioResources.bigVGANSafetensorsFilename,
        ] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: resources.bigVGANURL.appendingPathComponent(filename).path,
                contents: Data()
            ))
        }

        XCTAssertEqual(
            resources.bigVGANWeightsURL().lastPathComponent,
            MMAudioResources.bigVGANSafetensorsFilename
        )
    }

    func testManagedManifestValidatesAsNativeMMAudioWithoutDiffusersComponents() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mmaudio-manifest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = MMAudioModelResources(rootURL: root)
        try FileManager.default.createDirectory(
            at: resources.clipTokenizerURL,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: resources.bigVGANURL,
            withIntermediateDirectories: true
        )
        for url in [
            resources.networkWeightsURL,
            resources.clipWeightsURL,
            resources.synchformerWeightsURL,
            resources.vaeWeightsURL,
            resources.clipTokenizerURL.appendingPathComponent("tokenizer.json"),
            resources.bigVGANConfigURL,
            resources.bigVGANURL.appendingPathComponent(MMAudioResources.bigVGANPyTorchFilename),
        ] {
            XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
        }
        _ = try MereRunModelManifest.writeTemplateIfKnown(
            modelId: MMAudioResources.modelID,
            to: root
        )

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: MMAudioResources.modelID
        )
        XCTAssertEqual(report.errors, [])
        XCTAssertFalse(report.warnings.contains { $0.contains("engine mismatch") })
        XCTAssertFalse(report.warnings.contains { $0.contains("model root marker") })
    }
}
