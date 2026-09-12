import Foundation
import MereRunCore
import MereRunContract

extension APIServerContract {
    static let defaultVideoModelID = ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
    static let videoGenerationRoutePath = "/v1/videos/generations"

    static func decodeVideoGenerationRequest(from data: Data) throws -> OpenAIVideoGenerationRequest {
        try decodeJSONRequest(OpenAIVideoGenerationRequest.self, from: data)
    }

    struct VideoGenerationPlan: Equatable, Sendable {
        let modelID: String
        let prompt: String
        let width: Int
        let height: Int
        let seconds: Double?
        let numFrames: Int?
        let fps: Int
        let seed: Int?
        let quality: LTXVideoQuality?
        let outputMode: LTXVideoOutputMode?
        let options: [String]
    }

    static func videoGenerationPlan(
        from request: OpenAIVideoGenerationRequest
    ) throws -> VideoGenerationPlan {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        let modelID = normalizedOptional(request.model) ?? defaultVideoModelID
        let size = try videoSize(from: request.size)
        if let seconds = request.seconds,
           !seconds.isFinite || seconds <= 0 {
            throw APIRequestValidationError.invalidField("seconds", "must be finite and positive")
        }
        if let numFrames = request.num_frames, numFrames < 9 {
            throw APIRequestValidationError.invalidField("num_frames", "must be at least 9")
        }
        if request.seconds != nil, request.num_frames != nil {
            throw APIRequestValidationError.invalidField(
                "seconds",
                "use seconds or num_frames, not both"
            )
        }
        let fps = request.fps ?? 24
        guard fps > 0, fps <= 240 else {
            throw APIRequestValidationError.invalidField("fps", "must be between 1 and 240")
        }
        let quality: LTXVideoQuality?
        if let value = normalizedOptional(request.quality)?.lowercased() {
            guard let parsed = LTXVideoQuality(rawValue: value) else {
                throw APIRequestValidationError.invalidField("quality", "expected draft or final")
            }
            quality = parsed
        } else {
            quality = nil
        }
        let outputMode: LTXVideoOutputMode?
        if let value = normalizedOptional(request.output_mode)?.lowercased() {
            guard let parsed = LTXVideoOutputMode(rawValue: value) else {
                throw APIRequestValidationError.invalidField("output_mode", "expected video-only or audio-video")
            }
            outputMode = parsed
        } else {
            outputMode = nil
        }
        let options = try videoGenerationOptions(request.options ?? [])
        return VideoGenerationPlan(
            modelID: modelID,
            prompt: prompt,
            width: size.width,
            height: size.height,
            seconds: request.seconds,
            numFrames: request.num_frames,
            fps: fps,
            seed: request.seed,
            quality: quality,
            outputMode: outputMode,
            options: options
        )
    }

    static func videoGenerationResponse(
        outputURL: URL,
        plan: VideoGenerationPlan,
        createdAt: Date = Date()
    ) throws -> OpenAIVideoGenerationResponse {
        let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
        let exrDirectory = outputURL.deletingLastPathComponent().appendingPathComponent(
            outputURL.deletingPathExtension().lastPathComponent + "_exr",
            isDirectory: true
        )
        let hasEXR = FileManager.default.fileExists(atPath: exrDirectory.path)
        return OpenAIVideoGenerationResponse(
            created: Int(createdAt.timeIntervalSince1970),
            model: plan.modelID,
            artifact: OpenAIVideoGenerationArtifact(
                url: outputURL.absoluteString,
                media_type: "video/mp4",
                byte_count: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
                sha256: try ModelArtifactPin.fileSHA256(outputURL)
            ),
            exr_directory_url: hasEXR ? exrDirectory.absoluteString : nil
        )
    }

    private static func videoSize(from rawValue: String?) throws -> (width: Int, height: Int) {
        guard let rawValue = normalizedOptional(rawValue), rawValue.lowercased() != "auto" else {
            return (768, 512)
        }
        let parts = rawValue.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let width = Int(parts[0]),
              let height = Int(parts[1]),
              width >= 64,
              height >= 64,
              width <= 4_096,
              height <= 4_096,
              width.isMultiple(of: 64),
              height.isMultiple(of: 64),
              width <= 4_194_304 / height else {
            throw APIRequestValidationError.invalidField(
                "size",
                "expected WIDTHxHEIGHT with 64-pixel alignment and at most 4194304 pixels"
            )
        }
        return (width, height)
    }

    private static func videoGenerationOptions(_ options: [String]) throws -> [String] {
        guard options.count <= 256 else {
            throw APIRequestValidationError.invalidField("options", "must contain at most 256 arguments")
        }
        let protectedFlags: Set<String> = [
            "--output", "-o", "--model", "-m", "--width", "--height",
            "--duration", "--num-frames", "--fps", "--seed", "--quality",
            "--output-mode", "--preflight", "--json",
        ]
        var totalBytes = 0
        for option in options {
            totalBytes += option.utf8.count
            guard option.utf8.count <= 8_192, totalBytes <= 65_536 else {
                throw APIRequestValidationError.invalidField(
                    "options",
                    "arguments must total at most 65536 UTF-8 bytes"
                )
            }
            let flag = String(option.split(separator: "=", maxSplits: 1)[0])
            if flag == "--skip-mp4" {
                throw APIRequestValidationError.invalidField(
                    "options",
                    "--skip-mp4 is unavailable because this API route retains and hashes an MP4 artifact"
                )
            }
            if protectedFlags.contains(flag) {
                throw APIRequestValidationError.invalidField(
                    "options",
                    "\(flag) is controlled by a typed request field"
                )
            }
        }
        return options
    }
}
