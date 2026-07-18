import ArgumentParser
import Foundation
import AudioCore
import AudioSTT
import MereRunCore

#if os(macOS)
@preconcurrency import AVFoundation
import AudioToolbox
import CoreAudio
#endif

struct SpeechListen: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "listen",
        abstract: "Transcribe a macOS microphone with live Qwen ASR."
    )

    @Option(name: [.long], help: "CoreAudio input-device UID. Defaults to the system input.")
    var device: String?

    @Flag(name: [.long], help: "List input-device UIDs and exit.")
    var listDevices: Bool = false

    @Option(name: [.long], help: "Language hint (for example, en or zh).")
    var language: String?

    @Option(name: [.customShort("m"), .long], help: "Qwen model ID or local model path.")
    var model: String?

    @Option(name: [.customLong("decode-ms")], help: "Partial decode cadence in milliseconds.")
    var decodeMs: Int = 2_000

    @Option(name: [.customLong("silence-ms")], help: "Silence required to commit an utterance.")
    var silenceMs: Int = 900

    @Flag(name: [.short, .long], help: "Suppress diagnostics and interactive partials.")
    var quiet: Bool = false

    @Flag(name: [.long], help: "Emit versioned live ASR events as JSON Lines.")
    var jsonl: Bool = false

    func validate() throws {
        guard decodeMs > 0 else { throw ValidationError("--decode-ms must be > 0.") }
        guard silenceMs > 0 else { throw ValidationError("--silence-ms must be > 0.") }
    }

    func run() async throws {
        #if os(macOS)
        if listDevices {
            for input in try CoreAudioInputDevice.inputs() {
                print("\(input.isDefault ? "*" : " ") \(input.uid)\t\(input.name)")
            }
            return
        }
        guard await MicrophoneCapture.requestPermission() else {
            throw ValidationError("Microphone permission was denied. Enable it in System Settings > Privacy & Security > Microphone.")
        }
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let progressHandler: (@Sendable (ASRProgress) -> Void)?
        if quiet {
            progressHandler = nil
        } else {
            progressHandler = { progress in
                if let message = progress.message { CLIStderr.write("[\(progress.stage.rawValue)] \(message)\n") }
            }
        }
        let generator = try await CLIQwenASRLoader.prepare(model: model, progressHandler: progressHandler)
        let live = Qwen3ASRLiveSession(
            generator: generator,
            request: ASRStreamingRequest(
                language: language,
                sampleRate: 16_000,
                decodeIntervalMs: decodeMs,
                minDecodeAudioMs: 1_600
            ),
            configuration: Qwen3ASRLiveConfiguration(
                decodeIntervalMs: decodeMs,
                silenceMs: silenceMs
            )
        )
        if jsonl { try LiveASRCLIWriter.write(.ready()) }
        let writer = Task { try await LiveASRCLIWriter.consume(live.events, jsonl: jsonl, quiet: quiet) }
        let capture = try MicrophoneCapture(deviceUID: device)
        let stopSignal = InterruptSignal()
        do {
            try capture.start()
            if !quiet { CLIStderr.write("Listening. Press Ctrl-C to stop.\n") }
            let termination = try await withThrowingTaskGroup(of: ListenTermination.self) { group in
                group.addTask {
                    await stopSignal.wait()
                    return .interrupted
                }
                group.addTask {
                    for await samples in capture.samples {
                        try Task.checkCancellation()
                        try await live.feed(samples: samples)
                    }
                    try Task.checkCancellation()
                    return .captureEnded
                }
                let first = try await group.next() ?? .captureEnded
                group.cancelAll()
                return first
            }
            capture.stop()
            guard termination == .interrupted else {
                throw ValidationError("Microphone capture ended unexpectedly. Check that the selected input is still connected.")
            }
            try await live.finish(reason: .stopped)
            try await writer.value
        } catch {
            capture.stop()
            writer.cancel()
            await live.cancel()
            if jsonl {
                try? LiveASRCLIWriter.write(.error(
                    code: LiveASRCLIWriter.errorCode(for: error),
                    message: error.localizedDescription
                ))
            }
            throw error
        }
        #else
        throw ValidationError("mere.run speech listen requires the macOS live-capture backend.")
        #endif
    }
}

#if os(macOS)
private enum ListenTermination: Sendable {
    case interrupted
    case captureEnded
}

private struct CoreAudioInputDevice {
    let id: AudioDeviceID
    let uid: String
    let name: String
    let isDefault: Bool

    static func inputs() throws -> [Self] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else {
            throw ValidationError("Could not enumerate CoreAudio devices.")
        }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else {
            throw ValidationError("Could not read CoreAudio devices.")
        }
        let defaultID = defaultInputID()
        return ids.compactMap { id in
            guard hasInputChannels(id), let uid = stringProperty(id, kAudioDevicePropertyDeviceUID),
                  let name = stringProperty(id, kAudioObjectPropertyName) else { return nil }
            return Self(id: id, uid: uid, name: name, isDefault: id == defaultID)
        }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private static func defaultInputID() -> AudioDeviceID {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        _ = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id)
        return id
    }

    private static func hasInputChannels(_ id: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr,
              size >= MemoryLayout<AudioBufferList>.size else { return false }
        let storage = UnsafeMutableRawPointer.allocate(
            byteCount: Int(size),
            alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { storage.deallocate() }
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, storage) == noErr else { return false }
        let buffers = UnsafeMutableAudioBufferListPointer(
            storage.assumingMemoryBound(to: AudioBufferList.self)
        )
        return buffers.contains { $0.mNumberChannels > 0 }
    }

    private static func stringProperty(_ id: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let storage = UnsafeMutablePointer<CFString?>.allocate(capacity: 1)
        storage.initialize(to: nil)
        defer {
            storage.deinitialize(count: 1)
            storage.deallocate()
        }
        var size = UInt32(MemoryLayout<CFString?>.size)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, storage) == noErr,
              let value = storage.pointee else { return nil }
        return value as String
    }
}

private final class MicrophoneCapture: @unchecked Sendable {
    let samples: AsyncStream<[Float]>
    private let continuation: AsyncStream<[Float]>.Continuation
    private let engine = AVAudioEngine()
    private let converter: AVAudioConverter
    private let targetFormat: AVAudioFormat
    private let stateLock = NSLock()
    private var tapInstalled = false
    private var stopped = false
    private var configurationObserver: NSObjectProtocol?

    static func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    init(deviceUID: String?) throws {
        var captured: AsyncStream<[Float]>.Continuation?
        self.samples = AsyncStream { captured = $0 }
        self.continuation = captured!
        let input = engine.inputNode
        if let deviceUID {
            guard let selected = try CoreAudioInputDevice.inputs().first(where: { $0.uid == deviceUID }) else {
                throw ValidationError("No input device has CoreAudio UID '\(deviceUID)'. Use --list-devices.")
            }
            var id = selected.id
            guard let unit = input.audioUnit,
                  AudioUnitSetProperty(
                    unit,
                    kAudioOutputUnitProperty_CurrentDevice,
                    kAudioUnitScope_Global,
                    0,
                    &id,
                    UInt32(MemoryLayout<AudioDeviceID>.size)
                  ) == noErr else {
                throw ValidationError("Could not select input device '\(deviceUID)'.")
            }
        }
        let inputFormat = input.inputFormat(forBus: 0)
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: target) else {
            throw ValidationError("Could not configure 16 kHz mono microphone conversion.")
        }
        self.targetFormat = target
        self.converter = converter
    }

    func start() throws {
        let input = engine.inputNode
        let inputFormat = input.inputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1_600, format: inputFormat) { [weak self] buffer, _ in
            self?.convert(buffer)
        }
        stateLock.lock()
        tapInstalled = true
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: nil
        ) { [weak self] _ in
            self?.continuation.finish()
        }
        stateLock.unlock()
        engine.prepare()
        try engine.start()
    }

    func stop() {
        stateLock.lock()
        guard !stopped else {
            stateLock.unlock()
            return
        }
        stopped = true
        let removeTap = tapInstalled
        tapInstalled = false
        let observer = configurationObserver
        configurationObserver = nil
        stateLock.unlock()
        if let observer { NotificationCenter.default.removeObserver(observer) }
        if removeTap { engine.inputNode.removeTap(onBus: 0) }
        engine.stop()
        continuation.finish()
    }

    private func convert(_ input: AVAudioPCMBuffer) {
        let capacity = AVAudioFrameCount((Double(input.frameLength) * 16_000 / input.format.sampleRate).rounded(.up)) + 1
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }
        let inputState = MicrophoneConversionInput(buffer: input)
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, state in
            inputState.next(status: state)
        }
        guard status != .error, conversionError == nil, let channel = output.floatChannelData?[0] else {
            continuation.finish()
            return
        }
        continuation.yield(Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength))))
    }
}

private final class MicrophoneConversionInput: @unchecked Sendable {
    private let lock = NSLock()
    private let buffer: AVAudioPCMBuffer
    private var supplied = false

    init(buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        lock.lock()
        defer { lock.unlock() }
        guard !supplied else {
            status.pointee = .noDataNow
            return nil
        }
        supplied = true
        status.pointee = .haveData
        return buffer
    }
}

private final class InterruptSignal: @unchecked Sendable {
    private let stream: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation
    private let source: DispatchSourceSignal

    init() {
        var captured: AsyncStream<Void>.Continuation?
        stream = AsyncStream { captured = $0 }
        continuation = captured!
        signal(SIGINT, SIG_IGN)
        source = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        source.setEventHandler { [continuation] in
            continuation.yield(())
            continuation.finish()
        }
        source.resume()
    }

    func wait() async {
        for await _ in stream { break }
        source.cancel()
    }
}
#endif
