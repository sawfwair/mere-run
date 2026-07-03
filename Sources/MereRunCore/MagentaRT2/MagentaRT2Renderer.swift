import Foundation

#if canImport(Darwin)
import Darwin
#endif

#if canImport(magentart)
@preconcurrency import magentart
#endif

public struct MagentaRT2RenderRequest: Hashable, Sendable {
    public let prompt: String
    public let resources: MagentaRT2Resources
    public let durationSeconds: Float
    public let controls: MagentaRT2Controls

    public init(
        prompt: String,
        resources: MagentaRT2Resources,
        durationSeconds: Float,
        controls: MagentaRT2Controls = MagentaRT2Controls()
    ) {
        self.prompt = prompt
        self.resources = resources
        self.durationSeconds = durationSeconds
        self.controls = controls
    }
}

public struct MagentaRT2LiveControlSnapshot: Sendable {
    public let prompt: String?
    public let controls: MagentaRT2Controls?
    public let noteOn: [Int32]
    public let noteOff: [Int32]
    public let onsetMode: Int32?
    public let resetState: Bool
    public let shouldStop: Bool

    public init(
        prompt: String? = nil,
        controls: MagentaRT2Controls? = nil,
        noteOn: [Int32] = [],
        noteOff: [Int32] = [],
        onsetMode: Int32? = nil,
        resetState: Bool = false,
        shouldStop: Bool = false
    ) {
        self.prompt = prompt
        self.controls = controls
        self.noteOn = noteOn
        self.noteOff = noteOff
        self.onsetMode = onsetMode
        self.resetState = resetState
        self.shouldStop = shouldStop
    }
}

public enum MagentaRT2Renderer {
    public static func render(
        _ request: MagentaRT2RenderRequest,
        progress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> [Float] {
        guard request.durationSeconds > 0 else {
            throw MagentaRT2Error.invalidDuration(request.durationSeconds)
        }
        let missing = request.resources.validate()
        guard missing.isEmpty else {
            throw MagentaRT2Error.missingAssets(missing)
        }

        let frameCount = max(1, Int((request.durationSeconds * Float(MagentaRT2Resources.frameRate)).rounded(.up)))
        return try await renderFrames(request: request, frameCount: frameCount) { frameIndex, frame in
            progress?(frameIndex, frameCount)
            return frame.interleavedStereo
        }
    }

    public static func renderFrameStream(
        _ request: MagentaRT2RenderRequest,
        frameCount: Int,
        liveControls: (@Sendable (Int) throws -> MagentaRT2LiveControlSnapshot?)? = nil,
        onFrame: @escaping @Sendable (Int, MagentaRT2Frame) throws -> Void
    ) async throws {
        guard request.durationSeconds > 0 else {
            throw MagentaRT2Error.invalidDuration(request.durationSeconds)
        }
        let missing = request.resources.validate()
        guard missing.isEmpty else {
            throw MagentaRT2Error.missingAssets(missing)
        }
        _ = try await renderFrames(
            request: request,
            frameCount: max(1, frameCount),
            liveControls: liveControls
        ) { frameIndex, frame in
            try onFrame(frameIndex, frame)
            return []
        }
    }

    private static func renderFrames(
        request: MagentaRT2RenderRequest,
        frameCount: Int,
        liveControls: (@Sendable (Int) throws -> MagentaRT2LiveControlSnapshot?)? = nil,
        consumeFrame: @escaping @Sendable (Int, MagentaRT2Frame) throws -> [Float]
    ) async throws -> [Float] {
        #if canImport(magentart)
        return try await Task.detached(priority: .userInitiated) {
            try withSuppressedNativeStdout {
                guard let engine = mrt2_engine_create() else {
                    throw MagentaRT2Error.engineInitializationFailed(request.resources.resourcesURL.path)
                }
                defer { mrt2_engine_destroy(engine) }

                guard mrt2_engine_init_assets(engine, request.resources.resourcesURL.path, "musiccoca") else {
                    throw MagentaRT2Error.engineInitializationFailed(request.resources.resourcesURL.path)
                }
                guard mrt2_engine_load_model(engine, request.resources.modelURL.path) else {
                    throw MagentaRT2Error.modelLoadFailed(request.resources.modelURL.path)
                }

                applyControls(request.controls, to: engine)
                if request.controls.prefillSilence {
                    let frames = max(1, Int32(request.controls.prefillDurationSeconds * Float(MagentaRT2Resources.frameRate)))
                    guard mrt2_engine_prefill_silence(engine, frames) else {
                        throw MagentaRT2Error.engineInitializationFailed("silent prefill")
                    }
                }
                mrt2_engine_reset_state(engine)
                mrt2_engine_set_text_prompt(engine, request.prompt)
                try waitForPrompt(engine)

                var rendered: [Float] = []
                rendered.reserveCapacity(frameCount * MagentaRT2Resources.frameSamples * MagentaRT2Resources.channels)
                var left = [Float](repeating: 0, count: MagentaRT2Resources.frameSamples)
                var right = [Float](repeating: 0, count: MagentaRT2Resources.frameSamples)
                for frameIndex in 0..<frameCount {
                    try Task.checkCancellation()
                    if let snapshot = try liveControls?(frameIndex) {
                        if snapshot.shouldStop {
                            throw CancellationError()
                        }
                        if let controls = snapshot.controls {
                            applyControls(controls, to: engine)
                        }
                        if let onsetMode = snapshot.onsetMode {
                            mrt2_engine_set_onset_mode(engine, onsetMode)
                        }
                        snapshot.noteOff.forEach { mrt2_engine_set_note_off(engine, $0) }
                        snapshot.noteOn.forEach { mrt2_engine_set_note_on(engine, $0) }
                        if snapshot.resetState {
                            mrt2_engine_reset_state(engine)
                        }
                        if let prompt = snapshot.prompt {
                            // The wait is mandatory: overlapping
                            // mrt2_engine_generate_frame with the engine's
                            // asynchronous prompt encode segfaults (verified
                            // via MagentaRT2PromptSwapTests against the live
                            // engine), so a mid-session swap stalls the
                            // render loop for the encode. The engine's
                            // mrt2_runner_* API with its buffered audio ring
                            // is the path to stall-free swaps.
                            mrt2_engine_set_text_prompt(engine, prompt)
                            try waitForPrompt(engine)
                        }
                    }
                    let didGenerate = left.withUnsafeMutableBufferPointer { leftBuffer in
                        right.withUnsafeMutableBufferPointer { rightBuffer in
                            mrt2_engine_generate_frame(
                                engine,
                                leftBuffer.baseAddress,
                                rightBuffer.baseAddress,
                                Int32(MagentaRT2Resources.frameSamples)
                            )
                        }
                    }
                    guard didGenerate else {
                        throw MagentaRT2Error.generationFailed(frame: frameIndex)
                    }
                    let frame = MagentaRT2Frame(left: left, right: right)
                    rendered.append(contentsOf: try consumeFrame(frameIndex, frame))
                }
                return rendered
            }
        }.value
        #else
        _ = request
        _ = frameCount
        _ = consumeFrame
        throw MagentaRT2Error.unsupportedRuntime
        #endif
    }

    #if canImport(magentart)
    private static let nativeStdoutLock = NSLock()

    private static func applyControls(_ controls: MagentaRT2Controls, to engine: OpaquePointer) {
        mrt2_engine_set_musiccoca_token_count(engine, controls.styleConditioning.musicCoCaTokenCount)
        mrt2_engine_set_temperature(engine, controls.temperature)
        mrt2_engine_set_top_k(engine, controls.topK)
        mrt2_engine_set_cfg_musiccoca(engine, controls.cfgMusicCoCa)
        mrt2_engine_set_cfg_notes(engine, controls.cfgNotes)
        mrt2_engine_set_cfg_drums(engine, controls.cfgDrums)
        mrt2_engine_set_drumless(engine, controls.drumless)
        mrt2_engine_set_unmask_width(engine, controls.unmaskWidth)
        mrt2_engine_set_seed_rotation(engine, controls.seedRotation)
    }

    private static func waitForPrompt(_ engine: OpaquePointer) throws {
        let deadline = Date().addingTimeInterval(30)
        while mrt2_engine_get_text_encoder_status(engine) == 1 || mrt2_engine_get_quantizer_status(engine) == 1 {
            if Date() > deadline {
                throw MagentaRT2Error.promptEncodingFailed
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
        guard mrt2_engine_get_text_encoder_status(engine) != 3,
              mrt2_engine_get_quantizer_status(engine) != 3 else {
            throw MagentaRT2Error.promptEncodingFailed
        }
    }

    private static func withSuppressedNativeStdout<T>(_ body: () throws -> T) rethrows -> T {
        #if canImport(Darwin)
        nativeStdoutLock.lock()
        defer { nativeStdoutLock.unlock() }

        fflush(stdout)
        let savedStdout = dup(STDOUT_FILENO)
        guard savedStdout >= 0 else {
            return try body()
        }

        let devNull = open("/dev/null", O_WRONLY)
        guard devNull >= 0 else {
            close(savedStdout)
            return try body()
        }

        dup2(devNull, STDOUT_FILENO)
        close(devNull)
        defer {
            fflush(stdout)
            dup2(savedStdout, STDOUT_FILENO)
            close(savedStdout)
        }

        return try body()
        #else
        return try body()
        #endif
    }
    #endif
}
