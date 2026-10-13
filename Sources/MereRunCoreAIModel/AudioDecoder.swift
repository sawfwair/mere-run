// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
//
// Audio file decoding for the audio embedder: AVAudioFile (Apple's AudioToolbox decoders) at the file's own sample
// rate, as float32, every channel kept. The reference pipeline decodes with libsndfile (what upstream's librosa.load
// uses); Apple's decoder is sample-identical to it for PCM (WAV, AIFF, CAF), FLAC and ALAC files and within 3e-7 for
// AAC. For MP3 files Apple's decoder does not always honour the encoder delay and padding of the LAME/Xing/Info tag
// (whole frames for some files, one or two frames too many at the end for others), so `decode` brings its output to
// the tag's gapless length, after which the samples equal libsndfile's within 3e-5. An MP3 without that tag is
// returned as Apple decodes it (README "Audio: decoding").

import AVFoundation
import Foundation

public struct DecodedAudio: Sendable {
    public let sampleRate: Int
    public let channelCount: Int
    /// Non-interleaved: `channels[c][i]` is sample i of channel c (all channels have the same count).
    public let channels: [[Float]]
    public var frameCount: Int { channels.first?.count ?? 0 }

    public init(sampleRate: Int, channelCount: Int, channels: [[Float]]) {
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.channels = channels
    }
}

public enum AudioDecoder {
    /// Decodes a file (anything AVAudioFile reads: WAV, AIFF, CAF, FLAC, ALAC, AAC/M4A, MP3, ...) at its native rate.
    public static func decode(url: URL) throws -> DecodedAudio {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = file.processingFormat
        let channelCount = Int(format.channelCount)
        guard channelCount > 0 else { throw EmbedderError(description: "\(url.lastPathComponent): no audio channels") }
        var channels = [[Float]](repeating: [], count: channelCount)
        let chunk: AVAudioFrameCount = 1 << 16
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunk) else {
            throw EmbedderError(description: "\(url.lastPathComponent): cannot allocate a PCM buffer")
        }
        while true {
            try file.read(into: buffer, frameCount: chunk)
            let n = Int(buffer.frameLength)
            if n == 0 { break }
            guard let data = buffer.floatChannelData else { throw EmbedderError(description: "\(url.lastPathComponent): not float32 PCM") }
            for c in 0..<channelCount {
                channels[c].append(contentsOf: UnsafeBufferPointer(start: data[c], count: n))
            }
            if file.framePosition >= file.length { break }
        }
        if url.pathExtension.lowercased() == "mp3", let info = try? mp3GaplessInfo(url: url) {
            // AudioToolbox applies the tag itself for most files (afinfo reports "priming"/"remainder"), returns whole
            // frames for some, and for others trims the delay but keeps one or two whole frames too many at the end (it
            // counts the Xing/Info frame as audio: afinfo shows one packet more than the tag's frame count). Bring every
            // case to libsndfile's gapless length (same rule as the Python front end).
            let total = channels[0].count
            let full = info.frames * info.samplesPerFrame
            let gapless = full - info.delay - info.padding
            if total == full {
                let start = min(info.delay, total)
                let end = max(start, total - info.padding)
                channels = channels.map { Array($0[start..<end]) }
            } else if gapless > 0, total > gapless, (total - gapless) % info.samplesPerFrame == 0,
                      total - gapless <= 2 * info.samplesPerFrame {
                channels = channels.map { Array($0[0..<gapless]) }
            }
        }
        return DecodedAudio(sampleRate: Int(format.sampleRate.rounded()), channelCount: channelCount, channels: channels)
    }

    /// Frame count, samples per frame and the (encoder delay, padding) in samples from the Xing/Info + LAME tag of
    /// the first MPEG frame; nil without the tag or its frame count.
    public static func mp3GaplessInfo(url: URL) throws -> (frames: Int, samplesPerFrame: Int, delay: Int, padding: Int)? {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: 1 << 20) ?? Data()
        let b = [UInt8](data)
        var i = 0
        if b.count > 10, b[0] == 0x49, b[1] == 0x44, b[2] == 0x33 {   // "ID3"
            i = 10 + ((Int(b[6] & 0x7F) << 21) | (Int(b[7] & 0x7F) << 14) | (Int(b[8] & 0x7F) << 7) | Int(b[9] & 0x7F))
        }
        while i + 4 < b.count, !(b[i] == 0xFF && (b[i + 1] & 0xE0) == 0xE0) { i += 1 }
        guard i + 4 < b.count else { return nil }
        let mpeg1 = ((b[i + 1] >> 3) & 3) == 3
        let mono = ((b[i + 3] >> 6) & 3) == 3
        let side = mpeg1 ? (mono ? 17 : 32) : (mono ? 9 : 17)
        var p = i + 4 + side
        guard p + 8 <= b.count else { return nil }
        let tag = String(decoding: b[p..<(p + 4)], as: UTF8.self)
        guard tag == "Xing" || tag == "Info" else { return nil }
        let flags = (UInt32(b[p + 4]) << 24) | (UInt32(b[p + 5]) << 16) | (UInt32(b[p + 6]) << 8) | UInt32(b[p + 7])
        p += 8
        guard flags & 1 != 0, p + 4 <= b.count else { return nil }
        let frames = (Int(b[p]) << 24) | (Int(b[p + 1]) << 16) | (Int(b[p + 2]) << 8) | Int(b[p + 3])
        p += 4
        if flags & 2 != 0 { p += 4 }
        if flags & 4 != 0 { p += 100 }
        if flags & 8 != 0 { p += 4 }
        guard p + 24 <= b.count else { return nil }
        // the encoder tag ("LAME3.100", "Lavc63.1." ...) starts with ASCII letters; delay/padding are 12 bits each at +21
        guard b[p..<(p + 4)].allSatisfy({ ($0 >= 0x41 && $0 <= 0x5A) || ($0 >= 0x61 && $0 <= 0x7A) }) else { return nil }
        let d0 = Int(b[p + 21]), d1 = Int(b[p + 22]), d2 = Int(b[p + 23])
        return (frames, mpeg1 ? 1152 : 576, (d0 << 4) | (d1 >> 4), ((d1 & 0x0F) << 8) | d2)
    }
}

#endif
