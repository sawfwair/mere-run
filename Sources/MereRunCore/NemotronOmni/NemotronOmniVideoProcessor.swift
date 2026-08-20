import Foundation
import MediaIO
import MLX

struct NemotronOmniPreparedVideo: @unchecked Sendable {
    let pixelValues: MLXArray
    let frameCount: Int
    let tokensPerTubelet: Int
    let frameLabels: [String]

    var tubeletCount: Int {
        (frameCount + 1) / 2
    }

    var languageTokenCounts: [Int] {
        Array(repeating: tokensPerTubelet, count: tubeletCount)
    }

    var promptPrefix: String {
        (0..<tubeletCount).map { group in
            let first = group * 2 + 1
            let second = min(frameCount, first + 1)
            let firstLabel = frameLabels[first - 1]
            let label = first == second
                ? "Frame \(first)\(firstLabel): "
                : "Frame \(first)\(firstLabel) and frame \(second)\(frameLabels[second - 1]): "
            return label + "<image>"
        }.joined(separator: "\n") + "\n"
    }
}

enum NemotronOmniVideoProcessor {
    private static let framesPerSecond = 1.0

    static func prepare(
        reference: String,
        config: NemotronOmniPreprocessorConfig,
        contextLength: Int
    ) throws -> NemotronOmniPreparedVideo {
        let url = try localURL(reference: reference)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-nemotron-omni-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        // NVIDIA's published contract permits 128 frames at 1080p and 256
        // for lower-resolution sources. Probe with the bounded lower limit,
        // then use the decoded geometry to decide whether a second pass is
        // useful.
        var sequence = try MediaVideoIO.sampleFrames(
            from: url,
            into: directory,
            framesPerSecond: framesPerSecond,
            maximumFrames: NemotronOmniResources.maximumVideoFrames1080p
        )
        if max(sequence.frameWidth, sequence.frameHeight) < 1_080,
           sequence.frameURLs.count == NemotronOmniResources.maximumVideoFrames1080p {
            try FileManager.default.removeItem(at: directory)
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            sequence = try MediaVideoIO.sampleFrames(
                from: url,
                into: directory,
                framesPerSecond: framesPerSecond,
                maximumFrames: NemotronOmniResources.maximumVideoFrames720p
            )
        }
        guard !sequence.frameURLs.isEmpty else {
            throw NemotronOmniError.unsupportedMedia("video contains no decodable frames")
        }

        let prepared = try sequence.frameURLs.map { frameURL in
            try NemotronOmniImageProcessor.prepare(
                reference: frameURL.path,
                config: config,
                contextLength: contextLength,
                videoMode: true
            )
        }
        guard let first = prepared.first else {
            throw NemotronOmniError.unsupportedMedia("video contains no prepared frames")
        }
        guard prepared.allSatisfy({
            $0.pixelValues.dim(1) == first.pixelValues.dim(1)
                && $0.pixelValues.dim(2) == first.pixelValues.dim(2)
                && $0.sourcePatchCount == first.sourcePatchCount
        }) else {
            throw NemotronOmniError.unsupportedMedia(
                "video frame dimensions changed during decoding"
            )
        }
        let tokensPerTubelet = NemotronOmniPlaceholderPlanner.imageTokenCount(
            sourcePatchCount: first.sourcePatchCount
        )
        let totalTokens = ((prepared.count + 1) / 2) * tokensPerTubelet
        guard totalTokens + 64 <= contextLength else {
            throw NemotronOmniError.unsupportedMedia(
                "sampled video needs \(totalTokens) visual tokens, exceeding the \(contextLength)-token context"
            )
        }
        return NemotronOmniPreparedVideo(
            pixelValues: MLX.concatenated(prepared.map(\.pixelValues), axis: 0),
            frameCount: prepared.count,
            tokensPerTubelet: tokensPerTubelet,
            frameLabels: frameLabels(for: sequence)
        )
    }

    private static func frameLabels(for sequence: VideoFrameSequence) -> [String] {
        guard let indices = sequence.sourceFrameIndices,
              indices.count == sequence.frameURLs.count else {
            return Array(repeating: "", count: sequence.frameURLs.count)
        }
        let frameDurationMilliseconds = Int(1_000 / sequence.fps)
        return indices.map { index in
            let seconds = Double(index * frameDurationMilliseconds) / 1_000
            return String(format: " sampled at %.2f seconds", seconds)
        }
    }

    private static func localURL(reference: String) throws -> URL {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw NemotronOmniError.unsupportedMedia("video reference is empty")
        }
        if let parsed = URL(string: trimmed), parsed.scheme?.lowercased() == "file" {
            return parsed
        }
        if let scheme = URL(string: trimmed)?.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            throw NemotronOmniError.unsupportedMedia(
                "remote video URLs are not fetched; use a local path or file URL"
            )
        }
        let url = URL(fileURLWithPath: trimmed).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NemotronOmniError.unsupportedMedia("video file not found: \(url.path)")
        }
        return url
    }
}
