import Foundation
@testable import MereRunCore
import XCTest

final class DepthSequenceStreamingArtifactExporterTests: XCTestCase {
    func testStreamingExportMatchesLegacyArtifactsWithoutRetainingFrames() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let legacyRoot = root.appendingPathComponent("legacy", isDirectory: true)
        let streamingRoot = root.appendingPathComponent("streaming", isDirectory: true)
        let createdAt = Date(timeIntervalSince1970: 1_700_000_000)
        let inputURL = root.appendingPathComponent("input.mov")
        try Data("video-input".utf8).write(to: inputURL)
        let frames = try (0..<3).map { index in
            try DepthSequenceFrame(
                index: index,
                timeSeconds: Double(index) / 24,
                width: 2,
                height: 2,
                depth: [1, 2, 3, Float(index + 4)],
                confidence: [0.1, 0.2, 0.3, 0.4]
            )
        }
        let provenance = fixtureProvenance()
        let legacy = try DepthSequenceArtifactExporter.export(
            frames: frames,
            inputURL: inputURL,
            outputDirectory: legacyRoot,
            fps: 24,
            semantics: .affineRelative,
            provenance: provenance,
            createdAt: createdAt
        )
        let exporter = try DepthSequenceStreamingArtifactExporter(
            expectedFrameCount: frames.count,
            inputURL: inputURL,
            outputDirectory: streamingRoot,
            width: 2,
            height: 2,
            fps: 24,
            semantics: .affineRelative,
            provenance: provenance,
            createdAt: createdAt
        )
        for frame in frames { try exporter.append(frame) }
        XCTAssertEqual(exporter.spooledFrameCount, 3)
        let streaming = try exporter.finalize()

        XCTAssertEqual(streaming.manifest.frameCount, legacy.manifest.frameCount)
        XCTAssertEqual(streaming.manifest.inputByteCount, legacy.manifest.inputByteCount)
        XCTAssertEqual(streaming.manifest.inputSHA256, legacy.manifest.inputSHA256)
        XCTAssertEqual(streaming.manifest.frames.map(\.index), [0, 1, 2])
        XCTAssertEqual(streaming.manifest.frames.map(\.confidencePath), legacy.manifest.frames.map(\.confidencePath))
        XCTAssertEqual(
            streaming.manifest.frames.map { $0.artifacts.map(\.sha256) },
            legacy.manifest.frames.map { $0.artifacts.map(\.sha256) }
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: streaming.manifestURL.path))
    }

    func testRejectsOutOfOrderAndIncompleteSequences() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let exporter = try makeExporter(root: root, expectedFrameCount: 2, width: 1, height: 1)
        let outOfOrder = try DepthSequenceFrame(
            index: 1,
            timeSeconds: 0,
            width: 1,
            height: 1,
            depth: [1]
        )
        XCTAssertThrowsError(try exporter.append(outOfOrder)) { error in
            XCTAssertEqual(
                error as? DepthSequenceStreamingExporterError,
                .nonSequentialFrame(expected: 0, actual: 1)
            )
        }
        let first = try DepthSequenceFrame(
            index: 0,
            timeSeconds: 0,
            width: 1,
            height: 1,
            depth: [1]
        )
        try exporter.append(first)
        XCTAssertThrowsError(try exporter.finalize()) { error in
            XCTAssertEqual(
                error as? DepthSequenceStreamingExporterError,
                .incompleteSequence(expected: 2, actual: 1)
            )
        }
    }

    func testPreviewSamplingIsCappedIndependentlyOfSequenceLength() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let width = 1_000_001
        let exporter = try makeExporter(root: root, expectedFrameCount: 2, width: width, height: 1)
        let values = [Float](repeating: 1, count: width)
        for index in 0..<2 {
            try exporter.append(try DepthSequenceFrame(
                index: index,
                timeSeconds: Double(index),
                width: width,
                height: 1,
                depth: values
            ))
        }

        XCTAssertLessThanOrEqual(exporter.sampledDepthValueCount, 2_000_000)
        XCTAssertEqual(exporter.spooledFrameCount, 2)
        exporter.cancel()
    }

    private func makeExporter(
        root: URL,
        expectedFrameCount: Int,
        width: Int,
        height: Int
    ) throws -> DepthSequenceStreamingArtifactExporter {
        let inputURL = root.appendingPathComponent("input.mov")
        if !FileManager.default.fileExists(atPath: inputURL.path) {
            try Data("video-input".utf8).write(to: inputURL)
        }
        return try DepthSequenceStreamingArtifactExporter(
            expectedFrameCount: expectedFrameCount,
            inputURL: inputURL,
            outputDirectory: root.appendingPathComponent("output", isDirectory: true),
            width: width,
            height: height,
            fps: 24,
            semantics: .affineRelative,
            provenance: fixtureProvenance()
        )
    }

    private func fixtureProvenance() -> GeometryModelProvenance {
        GeometryModelProvenance(
            modelID: "fixture",
            upstreamRepository: "org/model",
            upstreamRevision: String(repeating: "a", count: 40),
            license: "Apache-2.0",
            weightsSHA256: String(repeating: "b", count: 64)
        )
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "depth-streaming-exporter-tests-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
