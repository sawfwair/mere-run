import Foundation
import XCTest
@testable import MereRunRelayKit

final class RelayEventTextTests: XCTestCase {
    func testNormalizationStripsSSEFramingAndSentinel() {
        let raw = """
        event: message
        id: 4
        data: {"sequence":0}

        data: {"sequence":1}
        data: [DONE]
        """
        XCTAssertEqual(
            RelayEventText.normalizedEventLines(raw),
            ["{\"sequence\":0}", "{\"sequence\":1}"]
        )
    }

    func testNormalizationPassesRawWorkerNDJSONThrough() {
        let raw = "{\"sequence\":0}\n{\"sequence\":1}\n"
        XCTAssertEqual(
            RelayEventText.normalizedEventLines(raw),
            ["{\"sequence\":0}", "{\"sequence\":1}"]
        )
    }

    func testDecodedEventsParsesEventLinesAndSkipsDiagnostics() throws {
        let event = GraphRunEvent(
            sequence: 3,
            createdAt: Date(timeIntervalSince1970: 1_700_000_000),
            type: "node_progress",
            state: .running,
            nodeID: "generate",
            message: "denoising",
            progress: GraphRunProgress(phase: "denoise", current: 2, total: 4, fraction: 0.5, unit: "steps")
        )
        let line = String(decoding: try WorkflowBundleCodec.lineEncoder().encode(event), as: UTF8.self)
        let raw = "data: \(line)\nnot-json diagnostic output\ndata: [DONE]\n"

        let decoded = RelayEventText.decodedEvents(raw)

        XCTAssertEqual(decoded, [event])
        XCTAssertEqual(decoded.first?.progress?.fraction, 0.5)
    }

    func testFileDigestMatchesInMemoryDigest() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("relaykit-digest-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        let payload = Data((0..<512).map { UInt8($0 % 251) })
        try payload.write(to: url)

        XCTAssertEqual(try ModelArtifactPinDigest.fileSHA256(url), ModelArtifactPinDigest.sha256(payload))
        XCTAssertEqual(try ModelArtifactPinDigest.fileByteCount(url), Int64(payload.count))
    }
}
