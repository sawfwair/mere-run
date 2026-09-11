import ArgumentParser
import Foundation
import MereRunCore

func parseMiniMaxH3ReferenceArguments(_ arguments: [String], requireFiles: Bool = true) throws -> [MiniMaxH3ReferenceInput] {
    try arguments.map { raw in
        guard let separator = raw.firstIndex(of: ":") else {
            throw ValidationError("--reference must be image:path, video:path, or audio:path (got \(raw)).")
        }
        let rawKind = String(raw[..<separator]).lowercased()
        let rawPath = String(raw[raw.index(after: separator)...])
        guard let kind = MiniMaxH3ReferenceKind(rawValue: rawKind), !rawPath.isEmpty else {
            throw ValidationError("--reference must be image:path, video:path, or audio:path (got \(raw)).")
        }
        let url = URL(fileURLWithPath: rawPath).standardizedFileURL
        guard !requireFiles || FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("Reference file not found: \(url.path)")
        }
        return MiniMaxH3ReferenceInput(kind: kind, url: url)
    }
}

func parseMiniMaxH3FrameArguments(_ arguments: [String], requireFiles: Bool = true) throws -> [MiniMaxH3FrameInput] {
    try arguments.map { raw in
        guard let separator = raw.firstIndex(of: ":") else {
            throw ValidationError("--h3-frame must be zero-based FRAME:PATH (got \(raw)).")
        }
        let rawIndex = String(raw[..<separator])
        let rawPath = String(raw[raw.index(after: separator)...])
        guard let frameIndex = Int(rawIndex), frameIndex >= 0, !rawPath.isEmpty else {
            throw ValidationError("--h3-frame must be zero-based FRAME:PATH (got \(raw)).")
        }
        let url = URL(fileURLWithPath: rawPath).standardizedFileURL
        guard !requireFiles || FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("H3 frame image not found: \(url.path)")
        }
        return MiniMaxH3FrameInput(frameIndex: frameIndex, url: url)
    }
}

struct VideoGenerationArgumentParser {
    let options: VideoGenerationOptions
    var fileManager: FileManager = .default
    var adaptersRoot: URL = MereRunModelPaths.adaptersDir
    var requireFiles = true

    func parseLTXImageConditionings() throws -> [LTXVideoConditioningInput] {
        try options.imageConditionings.map { raw in
            let parts = raw.split(separator: ":", maxSplits: 3, omittingEmptySubsequences: false)
            guard parts.count >= 2,
                  let frame = Int(parts[0]),
                  frame >= 0 else {
                throw ValidationError(
                    "--image-conditioning must be PIXEL_FRAME:PATH[:STRENGTH[:CRF]] (got \(raw))."
                )
            }
            let path = String(parts[1])
            guard !path.isEmpty else {
                throw ValidationError(
                    "--image-conditioning must include an image path (got \(raw))."
                )
            }
            let strength: Float
            if parts.count >= 3 {
                guard let parsed = Float(parts[2]), (0...1).contains(parsed) else {
                    throw ValidationError(
                        "--image-conditioning strength must be in [0, 1] (got \(parts[2]))."
                    )
                }
                strength = parsed
            } else {
                strength = 1
            }
            let crf: Int?
            if parts.count == 4 {
                guard let parsed = Int(parts[3]), (0...51).contains(parsed) else {
                    throw ValidationError("--image-conditioning CRF must be in 0...51 (got \(parts[3])).")
                }
                crf = parsed
            } else {
                crf = nil
            }
            let url = URL(fileURLWithPath: path).standardizedFileURL
            guard !requireFiles || fileManager.fileExists(atPath: url.path) else {
                throw ValidationError("Image conditioning file not found: \(url.path)")
            }
            return LTXVideoConditioningInput(
                imageURL: url,
                pixelFrameIndex: frame,
                strength: strength,
                crf: crf
            )
        }
    }

    func parseLTXLoRAConfigurations(
        _ arguments: [String],
        optionName: String,
        baseModelID: String
    ) throws -> [LTXLoRAConfiguration] {
        try arguments.map { raw in
            let separator = raw.lastIndex(of: "=")
            let path: String
            let strength: Float
            if let separator {
                path = String(raw[..<separator])
                let rawStrength = String(raw[raw.index(after: separator)...])
                guard let parsed = Float(rawStrength), parsed.isFinite else {
                    throw ValidationError(
                        "\(optionName) strength must be finite (got \(rawStrength))."
                    )
                }
                strength = parsed
            } else {
                path = raw
                strength = 1
            }
            guard !path.isEmpty else {
                throw ValidationError("\(optionName) must be PATH[=STRENGTH].")
            }
            let resolvedPath = try ManagedAdapterArgumentResolver.resolve(
                path,
                baseModelID: baseModelID,
                adaptersRoot: adaptersRoot,
                fileManager: fileManager,
                requireInstalled: requireFiles
            ) ?? path
            let url = URL(fileURLWithPath: resolvedPath).standardizedFileURL
            guard !requireFiles || fileManager.fileExists(atPath: url.path) else {
                throw ValidationError("LTX LoRA file not found: \(url.path)")
            }
            return LTXLoRAConfiguration(url: url, strength: strength)
        }
    }

    func parseLTXReferenceVideoConditionings(
        downscaleFactor: Int,
        temporalScaleFactor: Int
    ) throws -> [LTXReferenceVideoConditioningInput] {
        let attentionMaskURL = options.conditioningAttentionMask.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }
        if requireFiles, let attentionMaskURL,
           !fileManager.fileExists(atPath: attentionMaskURL.path) {
            throw ValidationError("IC-LoRA attention mask video not found: \(attentionMaskURL.path)")
        }
        return try options.videoConditionings.map { raw in
            let separator = raw.lastIndex(of: "=")
            let path: String
            let strength: Float
            if let separator {
                path = String(raw[..<separator])
                let rawStrength = String(raw[raw.index(after: separator)...])
                guard let parsed = Float(rawStrength), (0...1).contains(parsed) else {
                    throw ValidationError(
                        "--video-conditioning strength must be in [0, 1] (got \(rawStrength))."
                    )
                }
                strength = parsed
            } else {
                path = raw
                strength = 1
            }
            guard !path.isEmpty else {
                throw ValidationError("--video-conditioning must be PATH[=STRENGTH].")
            }
            let url = URL(fileURLWithPath: path).standardizedFileURL
            guard !requireFiles || fileManager.fileExists(atPath: url.path) else {
                throw ValidationError("IC-LoRA reference video not found: \(url.path)")
            }
            return LTXReferenceVideoConditioningInput(
                videoURL: url,
                strength: strength,
                attentionStrength: options.conditioningAttentionStrength == 1
                    ? nil
                    : options.conditioningAttentionStrength,
                attentionMaskVideoURL: attentionMaskURL,
                downscaleFactor: downscaleFactor,
                temporalScaleFactor: temporalScaleFactor
            )
        }
    }
}
