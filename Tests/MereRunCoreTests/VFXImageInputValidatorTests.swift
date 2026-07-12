import Foundation
import MediaIO
@testable import MereRunCore
import XCTest

final class VFXImageInputValidatorTests: MereRunCoreTestCase {
    func testInvalidPublicLimitsThrowInsteadOfTrapping() {
        let limits = VFXImageInputLimits(
            maximumDimension: 0,
            maximumPixelsPerImage: 10,
            maximumAggregatePixels: 5
        )
        XCTAssertThrowsError(try VFXImageInputValidator.validate(
            width: 1,
            height: 1,
            path: "frame.png",
            limits: limits
        )) { error in
            XCTAssertEqual(error as? VFXImageInputValidationError, .invalidLimits)
        }
    }

    func testValidatesDimensionPixelAndAggregateLimits() throws {
        XCTAssertEqual(
            try VFXImageInputValidator.validate(width: 640, height: 480, path: "frame.png"),
            VFXImageInputDimensions(width: 640, height: 480, pixelCount: 307_200)
        )

        let pixelLimits = VFXImageInputLimits(
            maximumDimension: 100,
            maximumPixelsPerImage: 5_000,
            maximumAggregatePixels: 20_000
        )
        XCTAssertThrowsError(try VFXImageInputValidator.validate(
            width: 100,
            height: 100,
            path: "frame.png",
            limits: pixelLimits
        )) { error in
            XCTAssertEqual(
                error as? VFXImageInputValidationError,
                .pixelLimitExceeded(path: "frame.png", pixels: 10_000, maximum: 5_000)
            )
        }

        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let imageURL = root.appendingPathComponent("aggregate.png")
        try writePNGAdvertising(width: 100, height: 100, to: imageURL)
        let aggregateLimits = VFXImageInputLimits(
            maximumDimension: 100,
            maximumPixelsPerImage: 10_000,
            maximumAggregatePixels: 15_000
        )
        XCTAssertThrowsError(try VFXImageInputValidator.inspectAndValidate(
            [imageURL, imageURL],
            limits: aggregateLimits
        )) { error in
            XCTAssertEqual(
                error as? VFXImageInputValidationError,
                .aggregatePixelLimitExceeded(pixels: 20_000, maximum: 15_000)
            )
        }
    }

    func testCompressedBombDimensionsAreRejectedFromHeaderBeforeDecode() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let imageURL = root.appendingPathComponent("compressed-bomb.png")
        try writeHighlyCompressedPNG(width: 50_000, height: 1, to: imageURL)
        let attributes = try FileManager.default.attributesOfItem(atPath: imageURL.path)
        let compressedBytes = (attributes[.size] as? NSNumber)?.intValue ?? .max
        let decodedRGBABytes = 50_000 * 4
        XCTAssertLessThan(compressedBytes * 100, decodedRGBABytes)
        let size = try MediaImageIO.size(of: imageURL)
        XCTAssertEqual(size.width, 50_000)
        XCTAssertEqual(size.height, 1)

        XCTAssertThrowsError(try VFXImageInputValidator.inspectAndValidate([imageURL])) { error in
            XCTAssertEqual(
                error as? VFXImageInputValidationError,
                .dimensionLimitExceeded(
                    path: imageURL.standardizedFileURL.path,
                    width: 50_000,
                    height: 1,
                    maximum: VFXImageInputLimits.production.maximumDimension
                )
            )
        }
    }

    func testReconstructionGeneratorsRejectCompressedBombBeforeCheckpointResolution() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let imageURL = root.appendingPathComponent("compressed-bomb.png")
        try writeHighlyCompressedPNG(width: 50_000, height: 1, to: imageURL)
        let expected = VFXImageInputValidationError.dimensionLimitExceeded(
            path: imageURL.standardizedFileURL.path,
            width: 50_000,
            height: 1,
            maximum: VFXImageInputLimits.production.maximumDimension
        )

        let moge = MoGe2Generator()
        do {
            _ = try await moge.generate(
                imageURL: imageURL,
                outputDirectory: root.appendingPathComponent("moge-output")
            )
            XCTFail("Expected MoGe-2 to reject the oversized image header")
        } catch {
            XCTAssertEqual(error as? VFXImageInputValidationError, expected)
        }

        let tripoSR = TripoSRGenerator()
        do {
            _ = try await tripoSR.generate(
                imageURL: imageURL,
                outputDirectory: root.appendingPathComponent("triposr-output")
            )
            XCTFail("Expected TripoSR to reject the oversized image header")
        } catch {
            XCTAssertEqual(error as? VFXImageInputValidationError, expected)
        }

        let instantMesh = InstantMeshGenerator()
        do {
            _ = try await instantMesh.generate(
                viewURLs: [imageURL, imageURL, imageURL, imageURL],
                outputDirectory: root.appendingPathComponent("instantmesh-output")
            )
            XCTFail("Expected InstantMesh to reject the oversized image header")
        } catch {
            XCTAssertEqual(error as? VFXImageInputValidationError, expected)
        }
    }

    func testImmutableSnapshotKeepsDecodedBytesAndProvenanceBoundTogether() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("replaceable.png")
        let original = try MediaImage(
            width: 2,
            height: 2,
            rgba8: Array(repeating: [UInt8(17), 17, 17, 255], count: 4).flatMap { $0 }
        )
        try MediaImageIO.writePNG(original, to: source)
        let expectedByteCount = try ModelArtifactPin.fileByteCount(source)
        let expectedSHA256 = try ModelArtifactPin.fileSHA256(source)

        let snapshot = try VFXImageInputSnapshotBatch.capture([source])
        defer { snapshot.cleanup() }
        try writeHighlyCompressedPNG(width: 50_000, height: 1, to: source)

        let decoded = try MediaImageIO.decode(try XCTUnwrap(snapshot.snapshotURLs.first))
        XCTAssertEqual(decoded.width, 2)
        XCTAssertEqual(decoded.height, 2)
        XCTAssertEqual(decoded.rgba8, original.rgba8)
        XCTAssertEqual(
            snapshot.inputRecords,
            [MeshInputRecord(
                path: source.standardizedFileURL.path,
                byteCount: expectedByteCount,
                sha256: expectedSHA256
            )]
        )
        XCTAssertNotEqual(try ModelArtifactPin.fileSHA256(source), expectedSHA256)
    }

    func testImmutableSnapshotBoundsEncodedBytesBeforeDecode() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("encoded-limit.png")
        try writeHighlyCompressedPNG(width: 2, height: 2, to: source)
        let limits = VFXImageInputLimits(
            maximumDimension: 10,
            maximumPixelsPerImage: 100,
            maximumAggregatePixels: 100,
            maximumEncodedBytesPerImage: 1,
            maximumAggregateEncodedBytes: 1
        )

        XCTAssertThrowsError(try VFXImageInputSnapshotBatch.capture([source], limits: limits)) {
            guard case .encodedByteLimitExceeded(let path, let bytes, let maximum) =
                $0 as? VFXImageInputValidationError else {
                return XCTFail("Unexpected error: \($0)")
            }
            XCTAssertEqual(path, source.standardizedFileURL.path)
            XCTAssertGreaterThan(bytes, 1)
            XCTAssertEqual(maximum, 1)
        }
    }

    func testGenericFileSnapshotUsesOneBoundedByteIdentity() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("clip.mov")
        try Data("original-video".utf8).write(to: source)

        let snapshot = try VFXFileInputSnapshot.capture(source, maximumEncodedBytes: 100)
        defer { snapshot.cleanup() }
        try Data("replacement".utf8).write(to: source)

        XCTAssertEqual(try Data(contentsOf: snapshot.snapshotURL), Data("original-video".utf8))
        XCTAssertEqual(snapshot.inputRecord.path, source.standardizedFileURL.path)
        XCTAssertEqual(snapshot.inputRecord.byteCount, 14)
        XCTAssertEqual(
            snapshot.inputRecord.sha256,
            try ModelArtifactPin.fileSHA256(snapshot.snapshotURL)
        )
        XCTAssertNotEqual(
            snapshot.inputRecord.sha256,
            try ModelArtifactPin.fileSHA256(source)
        )
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vfx-image-limit-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    /// Repeated transparent pixels make a valid PNG tiny on disk even though
    /// its decoded dimensions exceed the per-side admission limit.
    private func writePNGAdvertising(width: UInt32, height: UInt32, to url: URL) throws {
        try writeHighlyCompressedPNG(width: Int(width), height: Int(height), to: url)
    }

    private func writeHighlyCompressedPNG(width: Int, height: Int, to url: URL) throws {
        let image = try MediaImage(
            width: width,
            height: height,
            rgba8: [UInt8](repeating: 0, count: width * height * 4)
        )
        try MediaImageIO.writePNG(image, to: url)
    }
}
