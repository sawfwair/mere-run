import Foundation
import MLX
import MereRunResidency
import XCTest
@testable import AudioSTT
@testable import MereRunCore

private protocol QualifiedStreamOwner: Actor {
    func withRequestStream<Result>(_ operation: () async throws -> Result) async rethrows -> Result
    func unload() async
}

extension Gemma4Generator: QualifiedStreamOwner {}
extension ParakeetGenerator: QualifiedStreamOwner {}
extension LagunaGenerator: QualifiedStreamOwner {}

/// A reusable barrier proves that every lease is active before any operation can finish.
private actor StreamLeaseBarrier {
    private let width: Int
    private var arrivals: [CheckedContinuation<Void, Never>] = []

    init(width: Int) { self.width = width }

    func arrive() async {
        await withCheckedContinuation { continuation in
            arrivals.append(continuation)
            if arrivals.count == width {
                let ready = arrivals
                arrivals.removeAll()
                for waiter in ready { waiter.resume() }
            }
        }
    }
}

private actor StreamLeaseSignal {
    private var signaled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if signaled { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func signal() {
        signaled = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }
}

final class RequestStreamQualificationTests: MereRunCoreTestCase {
    private struct Identity: Hashable, Codable, Sendable {
        let cpu: String
        let gpu: String

        static func current() -> Self {
            Self(cpu: StreamOrDevice.cpu.stream.description, gpu: StreamOrDevice.gpu.stream.description)
        }
    }

    private enum InjectedFailure: Error { case afterSubmission }

    func testGemmaSequentialAndOverlappingLeases() async throws {
        try await checkReuse(Gemma4Generator(), name: "gemma")
    }

    func testParakeetSequentialAndOverlappingLeases() async throws {
        try await checkReuse(ParakeetGenerator(), name: "parakeet")
    }

    func testLagunaSequentialAndOverlappingLeases() async throws {
        try await checkReuse(LagunaGenerator(), name: "laguna")
    }

    func testLagunaCancellationErrorsAndUnload() async throws {
        try await checkRecovery(LagunaGenerator())
    }

    func testGemmaCancellationErrorsAndUnload() async throws {
        try await checkRecovery(Gemma4Generator())
    }

    func testParakeetCancellationErrorsAndUnload() async throws {
        try await checkRecovery(ParakeetGenerator())
    }

    func testRecreatedOwnersReuseProcessStreams() async throws {
        var gemma: Set<Identity> = []
        var parakeet: Set<Identity> = []
        var laguna: Set<Identity> = []
        for _ in 0..<10 {
            let chat = Gemma4Generator()
            gemma.insert(await chat.withRequestStream { Identity.current() })
            await chat.unload()
            let asr = ParakeetGenerator()
            parakeet.insert(await asr.withRequestStream { Identity.current() })
            await asr.unload()
            let nextChat = LagunaGenerator()
            laguna.insert(await nextChat.withRequestStream { Identity.current() })
            await nextChat.unload()
        }
        // Eviction can discard a generator. Sequential replacements must reuse
        // backend streams, including when switching between runtime families.
        XCTAssertEqual(gemma.count, 1)
        XCTAssertEqual(parakeet.count, 1)
        XCTAssertEqual(gemma, parakeet)
        XCTAssertEqual(gemma, laguna)
        try record(["gemma": Array(gemma), "parakeet": Array(parakeet), "laguna": Array(laguna)],
                   name: "recreated-owner-streams")
    }

    func testConcurrentRecreatedOwnersKeepExclusiveStreams() async throws {
        var all: Set<Identity> = []
        for _ in 0..<20 {
            let owners: [any QualifiedStreamOwner] = [
                Gemma4Generator(), ParakeetGenerator(), LagunaGenerator(), LagunaGenerator(),
            ]
            let barrier = StreamLeaseBarrier(width: owners.count)
            let identities = await withTaskGroup(of: Identity.self) { group in
                for owner in owners {
                    group.addTask {
                        await owner.withRequestStream {
                            let identity = Identity.current()
                            let value = MLXArray([Float(9)]) * 3
                            asyncEval(value)
                            await barrier.arrive()
                            XCTAssertEqual(Identity.current(), identity)
                            XCTAssertEqual(value.asArray(Float.self), [27])
                            return identity
                        }
                    }
                }
                var identities: Set<Identity> = []
                for await identity in group { identities.insert(identity) }
                return identities
            }
            XCTAssertEqual(identities.count, 4)
            all.formUnion(identities)
            XCTAssertEqual(all.count, 4)
            for owner in owners { await owner.unload() }
        }
        try record(["peakFourOwners": Array(all)], name: "recreated-concurrent-streams")
    }

    func testSelectedDeviceSurvivesCrossOwnerReuse() async throws {
        let chat = Gemma4Generator()
        let asr = ParakeetGenerator()
        var observed: [DeviceType: Set<Identity>] = [:]
        for _ in 0..<10 {
            for device in [DeviceType.cpu, .gpu] {
                for owner: any QualifiedStreamOwner in [chat, asr] {
                    let identity = await Device.withDefaultDevice(Device(device)) {
                        await owner.withRequestStream {
                            let identity = Identity.current()
                            XCTAssertEqual(StreamOrDevice.default.stream.description,
                                           device == .cpu ? identity.cpu : identity.gpu)
                            await Task.yield()
                            return identity
                        }
                    }
                    observed[device, default: []].insert(identity)
                }
            }
        }
        XCTAssertEqual(observed[.cpu]?.count, 1)
        XCTAssertEqual(observed[.gpu]?.count, 1)
        XCTAssertNotEqual(observed[.cpu], observed[.gpu])
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

    private func checkReuse<Owner: QualifiedStreamOwner>(_ owner: Owner, name: String) async throws {
        var sequential: Set<Identity> = []
        for index in 0..<200 {
            sequential.insert(await owner.withRequestStream {
                let identity = Identity.current()
                let cpu = MLX.multiply(MLXArray([Float(index)]), 2, stream: .cpu)
                let gpu = MLX.multiply(MLXArray([Float(index)]), 3, stream: .gpu)
                asyncEval(cpu, gpu)
                await Task.yield()
                XCTAssertEqual(Identity.current(), identity)
                XCTAssertEqual(cpu.asArray(Float.self), [Float(index * 2)])
                XCTAssertEqual(gpu.asArray(Float.self), [Float(index * 3)])
                return identity
            })
        }
        XCTAssertEqual(sequential.count, 1)
        var observed = sequential
        for width in [2, 4] {
            for _ in 0..<20 {
                let barrier = StreamLeaseBarrier(width: width)
                let identities = await withTaskGroup(of: Identity.self) { group in
                    for index in 0..<width {
                        group.addTask {
                            await owner.withRequestStream {
                                let identity = Identity.current()
                                let value = MLXArray([Float(index + 1)]) * 7
                                asyncEval(value)
                                await barrier.arrive()
                                XCTAssertEqual(Identity.current(), identity)
                                XCTAssertEqual(value.asArray(Float.self), [Float((index + 1) * 7)])
                                return identity
                            }
                        }
                    }
                    var result: [Identity] = []
                    for await identity in group { result.append(identity) }
                    return result
                }
                XCTAssertEqual(Set(identities).count, width)
                observed.formUnion(identities)
                XCTAssertEqual(observed.count, width)
            }
        }
        await owner.withRequestStream {
            let parent = Identity.current()
            await owner.withRequestStream {
                XCTAssertNotEqual(Identity.current(), parent)
                await Task.yield()
            }
            XCTAssertEqual(Identity.current(), parent)
        }
        try record(["sequential": Array(sequential), "peakFourLeases": Array(observed)], name: name + "-lease-streams")
    }

    private func checkRecovery<Owner: QualifiedStreamOwner>(_ owner: Owner) async throws {
        let initial = await owner.withRequestStream { Identity.current() }
        for submit in [false, true] {
            for _ in 0..<4 {
                do {
                    try await owner.withRequestStream {
                        if submit { asyncEval(MLXArray([Float(3)]) * 2) }
                        throw InjectedFailure.afterSubmission
                    }
                    XCTFail("Expected injected failure")
                } catch InjectedFailure.afterSubmission {
                    // Both pre-submission and post-submission errors must return the lease.
                }
                let next = await owner.withRequestStream { Identity.current() }
                XCTAssertEqual(next, initial)
            }
        }
        for phase in 0..<3 {
            for _ in 0..<4 {
                let entered = StreamLeaseSignal()
                let resume = StreamLeaseSignal()
                let task = Task {
                    try await owner.withRequestStream {
                        if phase > 0 { asyncEval(MLXArray([Float(7)]) * 2) }
                        if phase > 1 { await Task.yield() }
                        await entered.signal()
                        await resume.wait()
                        try Task.checkCancellation()
                    }
                }
                await entered.wait()
                task.cancel()
                await resume.signal()
                do {
                    try await task.value
                    XCTFail("Expected cooperative cancellation")
                } catch is CancellationError {
                    // Cancellation happens at a proven barrier, not after a guessed delay.
                }
                let next = await owner.withRequestStream { Identity.current() }
                XCTAssertEqual(next, initial)
            }
        }
        for _ in 0..<10 {
            await owner.unload()
            let next = await owner.withRequestStream { Identity.current() }
            XCTAssertEqual(next, initial)
        }
    }

    private func record(_ value: [String: [Identity]], name: String) throws {
        guard let output = ProcessInfo.processInfo.environment["MERERUN_STREAM_QUALIFICATION_OUTPUT"] else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = URL(fileURLWithPath: output).appendingPathComponent(name + ".json")
        try encoder.encode(value).write(to: url, options: .atomic)
    }
}
