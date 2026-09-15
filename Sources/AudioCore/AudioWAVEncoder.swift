import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct AudioWAVLayout {
    let dataBytes: Int
    let paddingBytes: Int
    let blockAlign: UInt16
    let byteRate: UInt32
    var fileBytes: Int { 44 + dataBytes + paddingBytes }

    init(sampleCount: Int, channels: Int, sampleRate: Int, encoding: AudioWAVEncoding) throws {
        guard (1...8).contains(channels) else { throw AudioExportError.invalidChannels(channels) }
        guard sampleRate > 0 else { throw AudioExportError.invalidSampleRate(sampleRate) }
        let width = Int(encoding.bitsPerSample / 8)
        guard sampleCount >= 0, sampleCount <= (Int(UInt32.max) - 36) / width else { throw AudioExportError.wavSizeLimit }
        dataBytes = sampleCount * width
        paddingBytes = dataBytes % 2
        guard dataBytes + paddingBytes <= Int(UInt32.max) - 36 else { throw AudioExportError.wavSizeLimit }
        blockAlign = UInt16(channels * width)
        guard sampleRate <= Int(UInt32.max) / Int(blockAlign) else { throw AudioExportError.wavSizeLimit }
        byteRate = UInt32(sampleRate * Int(blockAlign))
    }
}

public struct AudioExportData: Sendable {
    public let data: Data
    public let statistics: AudioExportStatistics
}

public struct AudioExportFile: Sendable {
    public let url: URL
    public let byteCount: Int
    public let statistics: AudioExportStatistics
}

public enum AudioWAVEncoder {
    public static func data(_ audio: ProcessedAudio) throws -> AudioExportData {
        try Task.checkCancellation()
        var data = Data()
        let layout = try layout(audio)
        data.reserveCapacity(layout.fileBytes)
        try chunks(audio, layout: layout) { data.append($0) }
        return AudioExportData(data: data, statistics: audio.statistics)
    }

    /// Encodes bounded chunks and publishes the completed file atomically.
    public static func write(_ audio: ProcessedAudio, to destination: URL) throws -> AudioExportFile {
        let layout = try layout(audio)
        try Task.checkCancellation()
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".audio-export-\(UUID().uuidString).tmp")
        let descriptor = temporary.path.withCString { open($0, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0o666) }
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close(); try? FileManager.default.removeItem(at: temporary) }
        try chunks(audio, layout: layout) { try handle.write(contentsOf: $0) }
        try handle.synchronize()
        try handle.close()
        try Task.checkCancellation()
        let renamed = temporary.path.withCString { source in destination.path.withCString { rename(source, $0) } }
        guard renamed == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        return AudioExportFile(url: destination, byteCount: layout.fileBytes, statistics: audio.statistics)
    }

    private static func layout(_ audio: ProcessedAudio) throws -> AudioWAVLayout {
        try AudioWAVLayout(sampleCount: audio.waveform.samples.count, channels: audio.waveform.channels,
                           sampleRate: audio.waveform.sampleRate, encoding: audio.plan.options.format)
    }

    private static func chunks(_ audio: ProcessedAudio, layout: AudioWAVLayout, sink: (Data) throws -> Void) throws {
        try Task.checkCancellation()
        try sink(header(layout: layout, channels: audio.waveform.channels,
                        sampleRate: audio.waveform.sampleRate, encoding: audio.plan.options.format))
        try payloadChunks(audio, sampleOffset: 0, sink: sink)
        try Task.checkCancellation()
        if layout.paddingBytes == 1 { try sink(Data([0])) }
    }

    static func header(layout: AudioWAVLayout, channels: Int, sampleRate: Int, encoding: AudioWAVEncoding) -> Data {
        var header = Data("RIFF".utf8)
        append(UInt32(36 + layout.dataBytes + layout.paddingBytes), to: &header)
        header.append(Data("WAVEfmt ".utf8))
        append(UInt32(16), to: &header)
        append(UInt16(encoding == .float32 ? 3 : 1), to: &header)
        append(UInt16(channels), to: &header)
        append(UInt32(sampleRate), to: &header)
        append(layout.byteRate, to: &header)
        append(layout.blockAlign, to: &header)
        append(encoding.bitsPerSample, to: &header)
        header.append(Data("data".utf8))
        append(UInt32(layout.dataBytes), to: &header)
        return header
    }

    static func payloadChunks(_ audio: ProcessedAudio, sampleOffset: Int, sink: (Data) throws -> Void) throws {
        let options = audio.plan.options
        let samples = audio.waveform.samples
        let chunkSamples = 16_384
        for start in stride(from: 0, to: samples.count, by: chunkSamples) {
            try Task.checkCancellation()
            let end = start + min(chunkSamples, samples.count - start)
            var chunk = Data()
            chunk.reserveCapacity((end - start) * Int(options.format.bitsPerSample / 8))
            for index in start..<end {
                let sample = samples[index]
                switch options.format {
                case .pcm16:
                    let noise = options.dither ? dither(sampleOffset + index) / 32_768 : 0
                    append(Int16(max(-1, min(1, sample + noise)) * 32_767), to: &chunk)
                case .pcm24:
                    let noise = options.dither ? dither(sampleOffset + index) / 8_388_608 : 0
                    let value = Int32(max(-1, min(1, sample + noise)) * 8_388_607)
                    chunk.append(UInt8(truncatingIfNeeded: value))
                    chunk.append(UInt8(truncatingIfNeeded: value >> 8))
                    chunk.append(UInt8(truncatingIfNeeded: value >> 16))
                case .float32:
                    append(sample.bitPattern, to: &chunk)
                }
            }
            try sink(chunk)
        }
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    private static func dither(_ index: Int) -> Float { random(UInt64(index) &* 2) - random(UInt64(index) &* 2 &+ 1) }
    private static func random(_ seed: UInt64) -> Float {
        var value = seed &+ 0x9E37_79B9_7F4A_7C15
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        value ^= value >> 31
        return Float(value & 0x00FF_FFFF) / Float(0x0100_0000)
    }
}

public enum AudioExportService {
    public static func data(_ waveform: AudioWaveform, plan: AudioExportPlan) throws -> AudioExportData {
        try AudioWAVEncoder.data(AudioExportProcessor.process(waveform, plan: plan))
    }
    public static func write(_ waveform: AudioWaveform, plan: AudioExportPlan, to url: URL) throws -> AudioExportFile {
        try AudioWAVEncoder.write(AudioExportProcessor.process(waveform, plan: plan), to: url)
    }
}
