#if !os(iOS)
import Foundation
import AudioCodecs
import MediaIO
import MLX
import MereRunQwenModel

struct ClefOmniMedia {
    struct Visual {
        let pixels: [Float]
        let temporal: Int
        let height: Int
        let width: Int
        var count: Int { temporal * height * width / 4 }

        func patches() -> MLXArray {
            MLXArray(pixels, [temporal, 2, 3, height / 2, 2, 16, width / 2, 2, 16])
                .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
                .reshaped(1, temporal * height * width, 3 * 2 * 16 * 16).asType(.bfloat16)
        }
    }
    struct Audio {
        let samples: [Float]
        var paddedSampleCount: Int?
        var frames: Int { max(1, min((samples.count + 159) / 160, (paddedSampleCount ?? samples.count) / 160)) }
        func mel() -> MLXArray {
            let padding = max(0, (paddedSampleCount ?? samples.count) - samples.count)
            return MelSpectrogram().extract(from: samples + Array(repeating: 0, count: padding))[0..., 0..., 0..<frames]
        }
        var count: Int { Self.tokenCount(frames: frames) }
        static func tokenCount(frames: Int) -> Int {
            // Three padded stride-2 convolutions reset at each 100-frame chunk.
            frames / 100 * 13 + (frames % 100 + 7) / 8
        }
    }
    struct Block {
        let text: String
        /// Payload kinds, and their local T/H/W positions. Video soundtrack tokens are interleaved by time.
        let kinds: [Int]
        let positions: [[Float]]
        let bosCount: Int
        let eosCount: Int
        let visual: Visual?
        let audio: Audio?
    }
    let blocks: [Block]
    var text: String { blocks.map(\.text).joined() }

    static func localURL(_ reference: String) throws -> URL {
        if let scheme = URL(string: reference)?.scheme, scheme != "file" {
            throw ClefError.invalidInput("Clef Omni media must use local files.")
        }
        let url: URL
        if reference.hasPrefix("file://") {
            guard let parsed = URL(string: reference), parsed.isFileURL else { throw ClefError.invalidInput("Invalid local media URL.") }
            url = parsed
        } else { url = URL(fileURLWithPath: reference) }
        guard FileManager.default.fileExists(atPath: url.path) else { throw ClefError.invalidInput("Media file not found: \(url.path)") }
        return url
    }

    static func prepare(_ request: ClefDecisionRequest) throws -> Self {
        var blocks: [Block] = []
        var reservedTokens = 0
        let temporary = FileManager.default.temporaryDirectory.appending(path: "clef-omni-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        func audio(_ reference: String) throws -> Audio {
            let decoded = try MediaAudioIO.decode(localURL(reference), targetSampleRate: 16_000, channels: 1)
            guard !decoded.samples.isEmpty, decoded.samples.count <= 384 * 16_000,
                  decoded.samples.allSatisfy(\.isFinite) else {
                throw ClefError.invalidInput("Clef Omni audio clips must contain finite samples and last at most 384 seconds.")
            }
            return Audio(samples: decoded.samples)
        }
        func visual(_ paths: [String], video: Bool, encodedVideo: Bool = false) throws -> Visual {
            func decode(_ path: String) throws -> MediaImage {
                let image = try MediaImageIO.decode(localURL(path))
                guard encodedVideo else { return image }
                let scale = min(1, sqrt(262_144 / Double(image.width * image.height)))
                let width = max(2, Int((Double(image.width) * scale / 2).rounded(.toNearestOrEven)) * 2)
                let height = max(2, Int((Double(image.height) * scale / 2).rounded(.toNearestOrEven)) * 2)
                return try MediaImageIO.bicubicResizedRGB(image, width: width, height: height)
            }
            let first = try decode(paths[0])
            let size = try Q35Generator.qwen3VLTargetSize(originalWidth: first.width, originalHeight: first.height, patchSize: 16, spatialMergeSize: 2,
                                                           minPixels: video ? 131_072 : 65_536,
                                                           maxPixels: video ? 786_432 : 1_048_576)
            let frameCount = video ? paths.count + paths.count % 2 : 2
            reservedTokens += frameCount / 2 * (size.height / 16) * (size.width / 16) / 4
            guard reservedTokens < request.maxTokens else { throw ClefError.invalidInput("Omni visual input exceeds the context budget.") }
            var paths = paths
            if !video || !paths.count.isMultiple(of: 2) { paths.append(paths.last!) }
            var pixels: [Float] = []
            for path in paths {
                try Task.checkCancellation()
                let image = try decode(path)
                guard image.width == first.width, image.height == first.height else {
                    throw ClefError.invalidInput("Frames in each Omni video must share dimensions.")
                }
                let resized = try MediaImageIO.bicubicResizedRGB(image, width: size.width, height: size.height)
                pixels += MediaImageIO.rescaledRGBCHWFloat(resized, rescaleFactor: Float(1 / 255.0), normalizedToMinusOneToOne: true)
            }
            return Visual(pixels: pixels, temporal: paths.count / 2, height: size.height / 16, width: size.width / 16)
        }
        for path in request.images {
            blocks.append(block(visual: try visual([path], video: false), audio: nil, video: false))
        }
        for path in request.audio { blocks.append(block(visual: nil, audio: try audio(path), video: false)) }
        let videoURLs = try request.videoFiles.map(localURL)
        let hearVideos = !videoURLs.isEmpty && videoURLs.allSatisfy(MediaVideoIO.hasAudioTrack)
        for (index, url) in videoURLs.enumerated() {
            let directory = temporary.appending(path: String(index))
            let sequence = try MediaVideoIO.sampleFrames(from: url, into: directory, framesPerSecond: 2,
                                                         maximumFrames: 769, strategy: .timestampFirstAtOrAfter)
            guard !sequence.frameURLs.isEmpty, sequence.frameURLs.count <= 768 else {
                throw ClefError.invalidInput("Clef Omni videos must contain 1–768 frames at 2 fps.")
            }
            blocks.append(block(visual: try visual(sequence.frameURLs.map(\.path), video: true, encodedVideo: true),
                                audio: hearVideos ? try audio(url.path) : nil, video: true))
        }
        for frames in request.videos {
            blocks.append(block(visual: try visual(frames, video: true), audio: nil, video: true))
        }
        // Whisper pads the entire audio batch before its centered STFT. Shorter clips
        // retain ceil(sampleCount / hop) frames, bounded by the longest clip's floor.
        let longestAudio = blocks.compactMap { $0.audio?.samples.count }.max()
        blocks = blocks.map { item in
            guard var audio = item.audio else { return item }
            audio.paddedSampleCount = longestAudio
            return block(visual: item.visual, audio: audio, video: item.kinds.contains(2))
        }
        guard blocks.reduce(0, { $0 + $1.kinds.count }) < request.maxTokens else {
            throw ClefError.invalidInput("Clef Omni media exceeds the context budget before schema and state.")
        }
        return Self(blocks: blocks)
    }

    static func block(visual: Visual?, audio: Audio?, video: Bool) -> Block {
        var visionPositions: [[Float]] = []
        if let visual {
            for temporal in 0..<visual.temporal {
                for height in 0..<(visual.height / 2) {
                    for width in 0..<(visual.width / 2) {
                        visionPositions.append([Float(temporal * 13), Float(height), Float(width)])
                    }
                }
            }
        }
        var kinds: [Int] = [], positions: [[Float]] = []
        var visionIndex = 0, audioIndex = 0
        let visionKind = video ? 2 : 1
        while visionIndex < visionPositions.count || audioIndex < (audio?.count ?? 0) {
            if visionIndex < visionPositions.count,
               audioIndex >= (audio?.count ?? 0) || visionPositions[visionIndex][0] <= Float(audioIndex) {
                kinds.append(visionKind); positions.append(visionPositions[visionIndex]); visionIndex += 1
            } else {
                kinds.append(3); positions.append(Array(repeating: Float(audioIndex), count: 3)); audioIndex += 1
            }
        }
        let visualBOS = visual == nil ? "" : "<|vision_start|>"
        let audioBOS = audio == nil ? "" : "<|audio_start|>"
        let visualEOS = visual == nil ? "" : "<|vision_end|>"
        let audioEOS = audio == nil ? "" : "<|audio_end|>"
        let payload = kinds.map { $0 == 1 ? "<|image_pad|>" : $0 == 2 ? "<|video_pad|>" : "<|audio_pad|>" }.joined()
        let markers = (visual == nil ? 0 : 1) + (audio == nil ? 0 : 1)
        return Block(text: visualBOS + audioBOS + payload + audioEOS + visualEOS + "\n",
                     kinds: kinds, positions: positions, bosCount: markers, eosCount: markers, visual: visual, audio: audio)
    }
}
#endif
