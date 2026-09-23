import ArgumentParser
import AudioSTT
import Foundation
import MereRunContract

enum Nemotron3LiveLatency: String, CaseIterable, ExpressibleByArgument {
    case standard = "1.04"
    case low = "0.64"
    case ultraLow = "0.32"

    var configuration: (chunk: Int, right: Int, fifo: Int, update: Int) {
        switch self {
        case .standard: Nemotron3DiarizationLatency.standard.configuration
        case .low: Nemotron3DiarizationLatency.low.configuration
        case .ultraLow: Nemotron3DiarizationLatency.ultraLow.configuration
        }
    }
}

struct SpeechDiarizeLive: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "diarize-live",
        abstract: "Stream Nemotron 3 speaker activity from a microphone or raw PCM stdin."
    )

    @Option(name: [.customShort("m"), .long], help: "Nemotron 3 managed model ID or local model directory.")
    var model: String = "speech-diarization-nemotron3"

    @Option(name: [.long], help: "CoreAudio input-device UID; defaults to the system input.")
    var device: String?

    @Flag(name: [.long], help: "List input-device UIDs and exit.")
    var listDevices = false

    @Flag(name: [.long], help: "Read 16 kHz mono signed 16-bit little-endian PCM from stdin.")
    var stdin = false

    @Option(name: [.long], help: "Input-buffer latency: 1.04, 0.64, or 0.32 seconds.")
    var latency: Nemotron3LiveLatency = .standard

    @Option(name: [.long], help: "Speaker activity threshold from 0 through 1.")
    var threshold: Float = 0.5

    @Flag(name: [.short, .long], help: "Suppress diagnostic output; JSON Lines remain on stdout.")
    var quiet = false

    func validate() throws {
        guard threshold.isFinite, (0...1).contains(threshold) else {
            throw ValidationError("--threshold must be between 0 and 1.")
        }
        guard !stdin || device == nil else {
            throw ValidationError("--device cannot be combined with --stdin.")
        }
    }

    func run() async throws {
        #if os(macOS)
        if listDevices {
            for input in try CoreAudioInputDevice.inputs() {
                print("\(input.isDefault ? "*" : " ") \(input.uid)\t\(input.name)")
            }
            return
        }
        if !stdin {
            guard await MicrophoneCapture.requestPermission() else {
                throw ValidationError(
                    "Microphone permission was denied. Enable it in System Settings > Privacy & Security > Microphone."
                )
            }
        }
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let modelRoot = try SpeechDiarize.resolveModelRoot(model)
        guard SpeechDiarize.isNemotron3(model: model, root: modelRoot) else {
            throw ValidationError("Live diarization requires speech-diarization-nemotron3.")
        }
        let diarizer = try Nemotron3Diarizer(modelDirectory: modelRoot)
        let setting = latency.configuration
        let inference = try diarizer.makeStreamingSession(
            threshold: threshold,
            chunkLength: setting.chunk,
            rightContext: setting.right,
            fifoLength: setting.fifo,
            cacheUpdatePeriod: setting.update
        )
        let live = LiveDiarizationSession(
            inference: inference, modelID: "speech-diarization-nemotron3",
            latency: latency.rawValue
        )
        try LiveDiarizationEventWriter.write(await live.ready(
            inputFormat: stdin ? "pcm_s16le" : "coreaudio_16khz_mono"
        ))
        do {
            if stdin {
                try await runStandardInput(live)
            } else {
                try await runMicrophone(live)
            }
        } catch {
            try? LiveDiarizationEventWriter.write(DiarizationStreamEvent(
                type: .error, code: "live_diarization_error", message: error.localizedDescription
            ))
            throw error
        }
        #else
        throw ValidationError("Live diarization requires the macOS capture and MLX runtime.")
        #endif
    }

    #if os(macOS)
    private func runStandardInput(_ live: LiveDiarizationSession) async throws {
        var decoder = PCM16LittleEndianDecoder()
        while let data = try FileHandle.standardInput.read(upToCount: 3_200), !data.isEmpty {
            let samples = decoder.decode(data)
            for event in try await live.feed(samples: samples) {
                try LiveDiarizationEventWriter.write(event)
            }
        }
        try decoder.validateEOF()
        for event in try await live.finish(reason: "eof") {
            try LiveDiarizationEventWriter.write(event)
        }
    }

    private func runMicrophone(_ live: LiveDiarizationSession) async throws {
        let capture = try MicrophoneCapture(deviceUID: device)
        let stopSignal = InterruptSignal()
        try capture.start()
        if !quiet { CLIStderr.write("Diarizing microphone audio. Press Ctrl-C to stop.\n") }
        do {
            let interrupted = try await withThrowingTaskGroup(of: Bool.self) { group in
                group.addTask {
                    await stopSignal.wait()
                    return true
                }
                group.addTask {
                    for await samples in capture.samples {
                        try Task.checkCancellation()
                        for event in try await live.feed(samples: samples) {
                            try LiveDiarizationEventWriter.write(event)
                        }
                    }
                    try Task.checkCancellation()
                    return false
                }
                let first = try await group.next() ?? false
                group.cancelAll()
                return first
            }
            capture.stop()
            guard interrupted else {
                throw ValidationError("Microphone capture ended unexpectedly. Check the input device.")
            }
            for event in try await live.finish(reason: "stopped") {
                try LiveDiarizationEventWriter.write(event)
            }
        } catch {
            capture.stop()
            throw error
        }
    }
    #endif
}
