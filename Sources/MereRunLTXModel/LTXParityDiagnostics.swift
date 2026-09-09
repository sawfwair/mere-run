import Foundation
import MLX

package func ltxAVDebugBaseURL() -> URL? {
    guard let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"],
          !debugPrefix.isEmpty else {
        return nil
    }
    return URL(fileURLWithPath: debugPrefix).standardizedFileURL
}

package func ltxAVDebugURL(suffix: String, fileExtension: String) -> URL? {
    guard let base = ltxAVDebugBaseURL() else { return nil }
    let parent = base.deletingLastPathComponent()
    try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    return parent.appendingPathComponent("\(base.lastPathComponent)_\(suffix).\(fileExtension)", isDirectory: false)
}

package func saveLTXAVDebugArray(_ array: MLXArray, suffix: String) {
    guard let url = ltxAVDebugURL(suffix: suffix, fileExtension: "npy") else { return }
    try? MLX.save(array: array, url: url)
}

package enum LTXAudioToVideoParityError: LocalizedError {
    case invalidNoiseShape(stage: String, expected: [Int], actual: [Int])

    package var errorDescription: String? {
        switch self {
        case .invalidNoiseShape(let stage, let expected, let actual):
            return "LTX A2Vid \(stage) parity noise has shape \(actual); expected \(expected)."
        }
    }
}

package enum LTXAudioToVideoNoiseStage: String {
    case stage1
    case stage2
}

package struct LTXAudioToVideoParityIO {
    package static let outputPrefixEnvironmentKey = "MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"
    package static let stage1NoiseEnvironmentKey = "MERERUN_VIDEO_LTX_A2VID_STAGE1_NOISE_PATH"
    package static let stage2NoiseEnvironmentKey = "MERERUN_VIDEO_LTX_A2VID_STAGE2_NOISE_PATH"

    package let outputPrefix: URL?
    package let stage1NoiseURL: URL?
    package let stage2NoiseURL: URL?

    package init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        self.outputPrefix = Self.url(environment[Self.outputPrefixEnvironmentKey])
        self.stage1NoiseURL = Self.url(environment[Self.stage1NoiseEnvironmentKey])
        self.stage2NoiseURL = Self.url(environment[Self.stage2NoiseEnvironmentKey])
    }

    package func resolveNoise(
        stage: LTXAudioToVideoNoiseStage,
        generated: MLXArray
    ) throws -> MLXArray {
        let sourceURL = switch stage {
        case .stage1: stage1NoiseURL
        case .stage2: stage2NoiseURL
        }
        guard let sourceURL else { return generated }

        let loaded = try MLX.loadArray(url: sourceURL)
        guard loaded.shape == generated.shape else {
            throw LTXAudioToVideoParityError.invalidNoiseShape(
                stage: stage.rawValue,
                expected: generated.shape,
                actual: loaded.shape
            )
        }
        return loaded.asType(generated.dtype)
    }

    package func save(_ array: MLXArray, suffix: String) throws {
        guard let outputPrefix else { return }
        let parent = outputPrefix.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let outputURL = parent.appendingPathComponent(
            "\(outputPrefix.lastPathComponent)_\(suffix).npy",
            isDirectory: false
        )
        try MLX.save(array: array, url: outputURL)
    }

    private static func url(_ rawValue: String?) -> URL? {
        guard let rawValue, !rawValue.isEmpty else { return nil }
        return URL(fileURLWithPath: rawValue).standardizedFileURL
    }
}
