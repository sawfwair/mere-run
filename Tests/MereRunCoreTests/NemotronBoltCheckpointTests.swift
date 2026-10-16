#if os(macOS)
import Darwin
import Foundation
import MLX
import MereRunAdmission
import XCTest
@testable import MereRunCore

/// Opt-in diagnostic suite over the original packed checkpoint, with externally
/// stored fixtures and raw records. This does not establish general model quality.
final class NemotronBoltCheckpointTests: MereRunCoreTestCase {
    private struct Manifest: Decodable {
        let cases: [Probe]
    }

    private struct Probe: Decodable {
        let id: String
        let prompt: String
        let image: String?
        let expected: String?
        let maxTokens: Int
        let thinking: Bool
    }

    private struct Record: Encodable {
        let id: String
        let prompt: String
        let image: String?
        let expected: String?
        let response: String
        let reasoning: String?
        let passed: Bool?
        let finishReason: String?
        let promptTokens: Int?
        let generatedTokens: Int
        let wallSeconds: Double
        let loadSeconds: Double?
        let prefillSeconds: Double?
        let decodeSeconds: Double?
        let timeToFirstPieceSeconds: Double?
        let mlxActiveBytes: Int
        let mlxPeakBytes: Int
        let processPeakRSSBytes: Int
        let physicalFootprintBytes: UInt64
        let peakPhysicalFootprintBytes: UInt64
    }

    private final class FirstPiece: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Date?
        func observe(_ progress: ChatProgress) {
            if progress.stage == .generating, progress.message?.isEmpty == false {
                lock.lock()
                if value == nil { value = Date() }
                lock.unlock()
            }
            if progress.stage == .loadingModel {
                let footprint = RuntimeMemorySample.current().physicalFootprintBytes ?? 0
                FileHandle.standardError.write(Data("[bolt-memory] mlx=\(Memory.activeMemory), peak=\(Memory.peakMemory), physical=\(footprint)\n".utf8))
            }
            if progress.stage != .generating {
                FileHandle.standardError.write(Data("[bolt] \(progress.message ?? "")\n".utf8))
            }
        }
        func seconds(since date: Date) -> Double? {
            lock.lock()
            defer { lock.unlock() }
            return value?.timeIntervalSince(date)
        }
    }

    func testRealCheckpointSuiteWhenProvided() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu",
              let root = env["MERERUN_TEST_BOLT_ROOT"],
              let suite = env["MERERUN_TEST_BOLT_SUITE"],
              let destination = env["MERERUN_TEST_BOLT_OUTPUT"] else {
            throw XCTSkip("Set GPU, MERERUN_TEST_BOLT_ROOT, MERERUN_TEST_BOLT_SUITE, and MERERUN_TEST_BOLT_OUTPUT.")
        }
        let suiteURL = URL(fileURLWithPath: suite)
        let output = URL(fileURLWithPath: destination)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: suiteURL))
        let selected = env["MERERUN_TEST_BOLT_CASES"]?.split(separator: ",").map(String.init)
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(coordinator.snapshot()).write(to: output.appendingPathComponent("admission-before.json"))
        let lease = try await coordinator.acquire(.init(label: "Nemotron Bolt real checkpoint qualification",
                                                       estimatedModelBytes: 24_079_678_824))
        defer { lease.release() }
        let generator = NemotronBoltGenerator()
        Memory.cacheLimit = 256 * 1_024 * 1_024
        for probe in manifest.cases where selected?.contains(probe.id) ?? true {
            Memory.clearCache()
            Memory.peakMemory = 0
            let first = FirstPiece()
            let start = Date()
            FileHandle.standardError.write(Data("[bolt-case] \(probe.id) begin\n".utf8))
            let response: ChatResponse
            do {
                response = try await generator.chat(ChatRequest(messages: [
                    ChatMessage(role: .user, content: probe.prompt,
                                imageUrl: probe.image.map { suiteURL.deletingLastPathComponent().appendingPathComponent($0).path })
                ], maxTokens: probe.maxTokens, temperature: 0, topP: 1, seed: 17,
                   showThinking: probe.thinking, maxContextTokens: 4_096), modelPath: root,
                   progressHandler: { first.observe($0) })
            } catch {
                let failure = "\(probe.id): \(error)\n"
                try Data(failure.utf8).write(to: output.appendingPathComponent("\(probe.id).error.txt"))
                await generator.unload()
                throw error
            }
            var footprint = rusage_info_v4()
            let footprintStatus = withUnsafeMutablePointer(to: &footprint) { pointer in
                pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
                }
            }
            XCTAssertEqual(footprintStatus, 0, "Kernel physical-footprint measurement failed")
            var usage = rusage()
            getrusage(RUSAGE_SELF, &usage)
            let visible = ChatReasoningMarkup.splitThinkBlocks(in: response.response).visibleContent
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let record = Record(id: probe.id, prompt: probe.prompt, image: probe.image,
                expected: probe.expected, response: response.response, reasoning: response.reasoningContent,
                passed: probe.expected.map { visible == $0 }, finishReason: response.finishReason?.rawValue,
                promptTokens: response.promptTokens, generatedTokens: response.tokensGenerated,
                wallSeconds: Date().timeIntervalSince(start), loadSeconds: response.timing?.loadSeconds,
                prefillSeconds: response.timing?.prefillSeconds, decodeSeconds: response.timing?.decodeSeconds,
                timeToFirstPieceSeconds: first.seconds(since: start), mlxActiveBytes: Memory.activeMemory,
                mlxPeakBytes: Memory.peakMemory, processPeakRSSBytes: usage.ru_maxrss,
                physicalFootprintBytes: footprint.ri_phys_footprint,
                peakPhysicalFootprintBytes: footprint.ri_lifetime_max_phys_footprint)
            try encoder.encode(record).write(to: output.appendingPathComponent("\(probe.id).json"), options: .atomic)
            FileHandle.standardError.write(Data("[bolt-case] \(probe.id): \(visible), \(record.wallSeconds)s, peakMLX=\(record.mlxPeakBytes)\n".utf8))
            XCTAssertFalse(response.response.isEmpty, probe.id)
            if probe.id == "smoke", record.passed != true {
                XCTFail("Released-checkpoint smoke failed; see the saved raw response")
                await generator.unload()
                return
            }
        }
        await generator.unload()
    }
}
#endif
