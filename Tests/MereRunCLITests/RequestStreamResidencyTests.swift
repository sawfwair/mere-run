import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import AudioSTT
@testable import MereRunCLI
@testable import MereRunCore

/// Serving owns runtime eviction; model tests do not depend on residency policy.
final class RequestStreamResidencyTests: MLXTestCase {
    private struct Identity: Hashable, Codable, Sendable {
        let cpu: String
        let gpu: String

        static func current() -> Self {
            Self(cpu: StreamOrDevice.cpu.stream.description, gpu: StreamOrDevice.gpu.stream.description)
        }
    }

    func testChatResidencyEvictionReusesStreamsAcrossGenerations() async throws {
        let cache = ResidentRuntimeCache<String, Gemma4Generator>(unload: { await $0.unload() })
        var identities: Set<Identity> = []
        var generations: Set<UUID> = []
        for _ in 0..<10 {
            let lease = try await cache.acquire(for: "gemma", make: { Gemma4Generator() }, prepare: { owner in
                await owner.withRequestStream { eval(MLXArray([Float(5)]) * 2) }
            })
            identities.insert(await lease.value.withRequestStream { Identity.current() })
            await lease.release()
            let snapshots = await cache.snapshots()
            let snapshot = try XCTUnwrap(snapshots["gemma"])
            generations.insert(snapshot.generation)
            let evicted = await cache.evictIfIdle(
                key: "gemma", generation: snapshot.generation, accessGeneration: snapshot.accessGeneration
            )
            XCTAssertTrue(evicted)
        }
        XCTAssertEqual(generations.count, 10)
        XCTAssertEqual(identities.count, 1)
        try record(["chatEviction": Array(identities)], name: "chat-eviction-streams")
    }

    func testASRResidencyReplacementAndEvictionReuseStreams() async throws {
        let slot = ResidentRuntimeSlot<Int, ParakeetGenerator>()
        var identities: Set<Identity> = []
        for index in 0..<10 {
            let identity = try await slot.withValue(
                for: index,
                make: { ParakeetGenerator() },
                unload: { await $0.unload() },
                operation: { owner in await owner.withRequestStream { Identity.current() } }
            )
            identities.insert(identity)
            if index % 2 == 1 {
                let evicted = await slot.evictIfIdle(expectedKey: index, reason: .ttl, using: { await $0.unload() })
                XCTAssertTrue(evicted)
            }
        }
        let state = await slot.state()
        XCTAssertEqual(state.loadCount, 10)
        XCTAssertEqual(state.replacementCount, 5)
        XCTAssertEqual(state.evictionCount, 5)
        XCTAssertEqual(identities.count, 1)
        try record(["asrReplacementAndEviction": Array(identities)], name: "asr-eviction-streams")
    }

    private func record(_ value: [String: [Identity]], name: String) throws {
        guard let output = ProcessInfo.processInfo.environment["MERERUN_STREAM_QUALIFICATION_OUTPUT"] else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = URL(fileURLWithPath: output).appendingPathComponent(name + ".json")
        try encoder.encode(value).write(to: url, options: .atomic)
    }
}
