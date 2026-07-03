import Foundation
import XCTest
@testable import MereRunCore

/// Live prompt-swap behavior against the real Magenta RT2 engine. Heavy
/// (loads the native engine plus a multi-GB model), so it only runs when
/// explicitly enabled:
///   MERERUN_TEST_RUN_MAGENTA=1 swift test --filter MagentaRT2PromptSwapTests
/// The model root defaults to the installed managed model and can be
/// overridden with MERERUN_TEST_MAGENTA_MODEL_ROOT.
///
/// History (2026-07-03): a non-blocking swap variant — keep calling
/// mrt2_engine_generate_frame while the engine encodes the new prompt on its
/// own thread — SEGFAULTED deterministically when run against the live
/// engine (signal 11 in the xctest process; the engine API is not safe for
/// cross-thread overlap with its asynchronous encode). The render loop
/// therefore blocks on prompt swaps by design; stall-free swaps need the
/// engine's threaded mrt2_runner_* API with its buffered audio ring. This
/// suite pins the supported blocking behavior.
final class MagentaRT2PromptSwapTests: XCTestCase {
    private func resolvedResources() throws -> MagentaRT2Resources {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_RUN_MAGENTA"] == "1" else {
            throw XCTSkip("Set MERERUN_TEST_RUN_MAGENTA=1 to run Magenta RT2 engine tests.")
        }
        let root: URL
        if let override = env["MERERUN_TEST_MAGENTA_MODEL_ROOT"], !override.isEmpty {
            root = URL(fileURLWithPath: override)
        } else {
            root = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support/MereRun/models/music-magenta-rt2-small")
        }
        guard FileManager.default.fileExists(atPath: root.path) else {
            throw XCTSkip("No Magenta RT2 model at \(root.path).")
        }
        let resources = MagentaRT2Resources(rootURL: root, modelID: "music-magenta-rt2-small")
        guard resources.validate().isEmpty else {
            throw XCTSkip("Magenta RT2 model at \(root.path) is missing assets.")
        }
        return resources
    }

    /// A mid-session prompt swap must complete the stream with finite,
    /// non-silent audio on both sides of the swap. The swap stalls the
    /// render loop for the encode by design (see the type comment), so
    /// there is deliberately no frame-time assertion around it.
    func testPromptSwapRendersToCompletion() async throws {
        let resources = try resolvedResources()
        let frameCount = 50
        let swapFrame = 20
        let recorder = FrameRecorder()

        let request = MagentaRT2RenderRequest(
            prompt: "gentle ambient piano",
            resources: resources,
            durationSeconds: Float(frameCount) / Float(MagentaRT2Resources.frameRate)
        )

        do {
            try await MagentaRT2Renderer.renderFrameStream(
                request,
                frameCount: frameCount,
                liveControls: { frameIndex in
                    if frameIndex == swapFrame {
                        return MagentaRT2LiveControlSnapshot(prompt: "energetic drum and bass")
                    }
                    return nil
                },
                onFrame: { frameIndex, frame in
                    recorder.record(frameIndex: frameIndex, frame: frame)
                }
            )
        } catch let error as MagentaRT2Error {
            throw XCTSkip("Magenta RT2 engine unavailable: \(error.localizedDescription)")
        }

        let stats = recorder.snapshot()
        XCTAssertEqual(stats.framesRendered, frameCount)
        XCTAssertTrue(stats.allFinite, "engine produced non-finite samples")
        XCTAssertGreaterThan(stats.peakAfterSwap, 0, "audio went silent after the prompt swap")
        print("[magenta-test] blocking swap completed, peak_after_swap=\(stats.peakAfterSwap)")
    }

    private final class FrameRecorder: @unchecked Sendable {
        struct Stats {
            let framesRendered: Int
            let allFinite: Bool
            let peakAfterSwap: Float
        }

        private let lock = NSLock()
        private var framesRendered = 0
        private var allFinite = true
        private var peakAfterSwap: Float = 0

        func record(frameIndex: Int, frame: MagentaRT2Frame) {
            lock.lock()
            defer { lock.unlock() }
            framesRendered += 1
            let peak = frame.left.reduce(Float(0)) { max($0, abs($1)) }
            if frame.left.contains(where: { !$0.isFinite }) || frame.right.contains(where: { !$0.isFinite }) {
                allFinite = false
            }
            if frameIndex >= 20 {
                peakAfterSwap = max(peakAfterSwap, peak)
            }
        }

        func snapshot() -> Stats {
            lock.lock()
            defer { lock.unlock() }
            return Stats(
                framesRendered: framesRendered,
                allFinite: allFinite,
                peakAfterSwap: peakAfterSwap
            )
        }
    }
}
