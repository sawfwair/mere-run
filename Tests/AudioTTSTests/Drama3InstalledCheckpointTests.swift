import AudioCore
import AudioDrama3Model
import AudioTTS
import Foundation
import MLX
import MereRunAdmission
import MereRunCore
import XCTest

/// Opt-in released-checkpoint probes. Outputs remain outside the repository.
final class Drama3InstalledCheckpointTests: XCTestCase {
    private func paths() throws -> (root: URL, evidence: URL) {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_TEST_DRAMA3_ROOT"],
              let evidence = environment["MERERUN_TEST_DRAMA3_EVIDENCE"] else {
            throw XCTSkip("Set MERERUN_TEST_DRAMA3_ROOT and MERERUN_TEST_DRAMA3_EVIDENCE for checkpoint qualification.")
        }
        let output = URL(fileURLWithPath: evidence)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        return (URL(fileURLWithPath: root), output)
    }

    func testReleasedCodecDeterministicDecodeAndEncode() throws {
        let paths = try paths()
        let codec = try Drama3Resources.loadCodec(root: paths.root)
        let codes: [[Int32]] = (0..<10).map { book -> [Int32] in
            let size = book == 0 ? 4096 : 1024
            return (0..<32).map { frame in Int32((frame * 73 + book * 19) % size) }
        }
        let decoded = try codec.decode(codes)
        MLX.eval(decoded)
        XCTAssertEqual(decoded.size, 32 * Drama3Codec.samplesPerFrame)
        XCTAssertTrue(decoded.asArray(Float.self).allSatisfy { $0.isFinite })
        try MLX.save(array: decoded, url: paths.evidence.appendingPathComponent("codec-native-decode.npy"))
        let samples = (0..<44_100).map { Float(0.1 * sin(Double($0) * 2 * .pi * 220 / 44_100)) }
        let encoded = try codec.encode(samples)
        try JSONEncoder().encode(encoded).write(to: paths.evidence.appendingPathComponent("codec-native-encode.json"))
        Memory.clearCache()
    }

    func testReleasedCheckpointSpeechAndMemory() async throws {
        let paths = try paths()
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let lease = try await coordinator.acquire(.init(label: "DRAMA-3 checkpoint qualification", resourceClass: .standard),
            onWait: { _ in
                FileHandle.standardError.write(Data("DRAMA-3 qualification is waiting for shared inference capacity.\n".utf8))
            })
        defer { lease.release() }
        let generator = Drama3Generator()
        let text = "[excited] Hello! The native speech engine is running on this laptop."
        let request = TTSRequest(text: text, voiceDescription: "", temperature: 0.6, seed: 42,
            outputURL: paths.evidence.appendingPathComponent("native-speech.wav"))
        let started = Date()
        let trace = MemoryTrace()
        let traceURL = paths.evidence.appendingPathComponent("native-speech-stages.json")
        #if os(macOS)
        GPU.resetPeakMemory()
        #endif
        do {
            let audio = try await generator.generateAudio(request, modelPath: paths.root.path, progressHandler: { progress in
                trace.append(progress)
                print("DRAMA-3 checkpoint: \(progress.stage.rawValue) frames=\(progress.tokensGenerated) "
                    + "active=\(Memory.activeMemory) cache=\(Memory.cacheMemory)")
            })
            try trace.write(to: traceURL)
            let decoding = try XCTUnwrap(trace.snapshot().first { $0.stage == TTSStage.decoding.rawValue })
            XCTAssertLessThan(decoding.activeMLXBytes, 1_073_741_824,
                "Batch generation must release the text/tower checkpoint before codec decoding.")
            XCTAssertGreaterThan(audio.duration, 0.5)
            XCTAssertTrue(audio.samples.allSatisfy(\.isFinite))
            let rms = sqrt(audio.samples.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(audio.samples.count))
            XCTAssertGreaterThan(rms, 0.0001)
            _ = try AudioExportService.write(audio, plan: SpeechSynthesisPlan(request: request).exportPlan, to: request.outputURL)
            let report = SpeechReport(text: text, seed: 42, elapsedSeconds: Date().timeIntervalSince(started),
                durationSeconds: audio.duration, sampleRate: audio.sampleRate, rms: rms,
                peakMLXBytes: Memory.peakMemory, activeMLXBytes: Memory.activeMemory,
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                device: Device.defaultDevice().deviceType?.rawValue ?? "unknown")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: paths.evidence.appendingPathComponent("native-speech-memory.json"))
            print("DRAMA-3 checkpoint: \(audio.duration)s audio; peak MLX \(Memory.peakMemory) bytes")
            await generator.unload()
        } catch {
            try trace.write(to: traceURL)
            await generator.unload()
            throw error
        }
    }

    private struct StageMemory: Encodable {
        let stage: String
        let frames: Int
        let activeMLXBytes: Int
        let cachedMLXBytes: Int
        let peakMLXBytes: Int
        let physicalFootprintBytes: UInt64?
    }

    private final class MemoryTrace: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [StageMemory] = []

        func append(_ progress: TTSProgress) {
            let event = StageMemory(stage: progress.stage.rawValue, frames: progress.tokensGenerated,
                activeMLXBytes: Memory.activeMemory, cachedMLXBytes: Memory.cacheMemory,
                peakMLXBytes: Memory.peakMemory,
                physicalFootprintBytes: RuntimeMemorySample.current().physicalFootprintBytes)
            lock.lock()
            defer { lock.unlock() }
            events.append(event)
        }

        func snapshot() -> [StageMemory] {
            lock.lock()
            defer { lock.unlock() }
            return events
        }

        func write(to url: URL) throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(snapshot()).write(to: url)
        }
    }

    private struct SpeechReport: Encodable {
        let text: String
        let seed: UInt64
        let elapsedSeconds: Double
        let durationSeconds: Double
        let sampleRate: Int
        let rms: Double
        let peakMLXBytes: Int
        let activeMLXBytes: Int
        let physicalMemoryBytes: UInt64
        let device: String
    }
}
