import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Encodes incremental audio with the shared processor and WAV encoder.
/// The caller serializes access. Only `finish()` publishes the destination.
public final class AudioExportStream {
    private let plan: AudioExportPlan
    private let destination: URL
    private let temporary: URL
    private let sampleRate: Int
    private let channels: Int
    private var handle: FileHandle?
    private var sampleCount = 0
    private var replacements = 0
    private var clippedSamples = 0
    private var inputPeak: Float = 0
    private var outputPeak: Float = 0
    private var sumSquares = 0.0

    public init(plan: AudioExportPlan, to destination: URL, sampleRate: Int, channels: Int = 1) throws {
        try Task.checkCancellation()
        guard plan.options.normalization == .none,
              plan.options.fadeInMilliseconds == 0, plan.options.fadeOutMilliseconds == 0 else {
            throw AudioExportError.unsupportedStreamingProcessing
        }
        let layout = try AudioWAVLayout(sampleCount: 0, channels: channels, sampleRate: sampleRate, encoding: plan.options.format)
        self.plan = plan
        self.destination = destination
        self.sampleRate = sampleRate
        self.channels = channels
        temporary = destination.deletingLastPathComponent().appendingPathComponent(".audio-export-\(UUID().uuidString).tmp")
        let descriptor = temporary.path.withCString { open($0, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0o666) }
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        handle = file
        do {
            try file.write(contentsOf: AudioWAVEncoder.header(layout: layout, channels: channels, sampleRate: sampleRate, encoding: plan.options.format))
        } catch {
            cancel()
            throw error
        }
    }

    deinit { cancel() }

    public func append(samples: [Float]) throws {
        do {
            try appendPayload(samples)
        } catch {
            cancel()
            throw error
        }
    }

    private func appendPayload(_ samples: [Float]) throws {
        guard let handle else { throw AudioExportError.streamClosed }
        try Task.checkCancellation()
        let (count, overflow) = sampleCount.addingReportingOverflow(samples.count)
        guard !overflow else { throw AudioExportError.wavSizeLimit }
        _ = try AudioWAVLayout(sampleCount: count, channels: channels, sampleRate: sampleRate, encoding: plan.options.format)
        let waveform = try AudioWaveform(interleaved: samples, channels: channels, sampleRate: sampleRate)
        let processed = try AudioExportProcessor.process(waveform, plan: plan)
        try AudioWAVEncoder.payloadChunks(processed, sampleOffset: sampleCount) { try handle.write(contentsOf: $0) }
        sampleCount = count
        replacements += processed.statistics.replacedNonfiniteSamples
        clippedSamples += processed.statistics.clippedSamples
        inputPeak = max(inputPeak, processed.statistics.inputPeak)
        outputPeak = max(outputPeak, processed.statistics.outputPeak)
        sumSquares += pow(processed.statistics.outputRMS, 2) * Double(samples.count)
    }

    public func finish() throws -> AudioExportFile {
        guard let handle else { throw AudioExportError.streamClosed }
        do {
            try Task.checkCancellation()
            let layout = try AudioWAVLayout(sampleCount: sampleCount, channels: channels, sampleRate: sampleRate, encoding: plan.options.format)
            if layout.paddingBytes == 1 { try handle.write(contentsOf: Data([0])) }
            try handle.seek(toOffset: 0)
            try handle.write(contentsOf: AudioWAVEncoder.header(layout: layout, channels: channels, sampleRate: sampleRate, encoding: plan.options.format))
            try handle.synchronize()
            try handle.close()
            self.handle = nil
            try Task.checkCancellation()
            let renamed = temporary.path.withCString { source in destination.path.withCString { rename(source, $0) } }
            guard renamed == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            return AudioExportFile(url: destination, byteCount: layout.fileBytes, statistics: AudioExportStatistics(
                sampleCount: sampleCount, replacedNonfiniteSamples: replacements, clippedSamples: clippedSamples,
                fadeInFrames: 0, fadeOutFrames: 0, normalizationGain: 1,
                inputPeak: inputPeak, outputPeak: outputPeak,
                outputRMS: sampleCount == 0 ? 0 : sqrt(sumSquares / Double(sampleCount))
            ))
        } catch {
            cancel()
            throw error
        }
    }

    /// Discards only this stream's temporary file, preserving an existing destination.
    public func cancel() {
        try? handle?.close()
        handle = nil
        try? FileManager.default.removeItem(at: temporary)
    }
}
