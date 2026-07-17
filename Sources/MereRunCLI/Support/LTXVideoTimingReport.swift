import Foundation
import MereRunCore

@inline(__always)
func videoMonotonicSeconds() -> Double {
    ProcessInfo.processInfo.systemUptime
}

struct LTXVideoTimingReport: Codable, Hashable, Sendable {
    let mode: String
    let modelRoot: String
    let residentModelReused: Bool
    let load: LTXLoadTimings
    let generation: LTXGenerationTimings
    let unloadSeconds: Double
    let mp4WriteSeconds: Double
    let totalSeconds: Double
}

func emitLTXVideoTimingReport(
    _ report: LTXVideoTimingReport,
    printToStandardError: Bool,
    outputPath: String?
) throws {
    if printToStandardError {
        CLIStderr.write("LTX timings:\n")
        CLIStderr.write("  load total: \(formatLTXSeconds(report.load.totalSeconds))s\n")
        CLIStderr.write("    text encoder: \(formatLTXSeconds(report.load.textEncoderSeconds))s\n")
        CLIStderr.write("    transformer: \(formatLTXSeconds(report.load.transformerSeconds))s\n")
        CLIStderr.write("    video decoder: \(formatLTXSeconds(report.load.videoDecoderSeconds))s\n")
        CLIStderr.write("    upsampler: \(formatLTXSeconds(report.load.upsamplerSeconds))s\n")
        CLIStderr.write("    audio decoder: \(formatLTXSeconds(report.load.audioDecoderSeconds))s\n")
        CLIStderr.write("  generation total: \(formatLTXSeconds(report.generation.totalSeconds))s\n")
        CLIStderr.write("    text encoding: \(formatLTXSeconds(report.generation.textEncodingSeconds))s\n")
        CLIStderr.write("    preparation: \(formatLTXSeconds(report.generation.preparationSeconds))s\n")
        CLIStderr.write("    stage 1 denoise: \(formatLTXSeconds(report.generation.stage1DenoiseSeconds))s\n")
        CLIStderr.write("    LoRA fusion: \(formatLTXSeconds(report.generation.loraFusionSeconds))s\n")
        CLIStderr.write("    upsample: \(formatLTXSeconds(report.generation.upsampleSeconds))s\n")
        CLIStderr.write("    stage 2 denoise: \(formatLTXSeconds(report.generation.stage2DenoiseSeconds))s\n")
        CLIStderr.write("    video decode: \(formatLTXSeconds(report.generation.videoDecodeSeconds))s\n")
        CLIStderr.write("    audio decode: \(formatLTXSeconds(report.generation.audioDecodeSeconds))s\n")
        CLIStderr.write("  unload: \(formatLTXSeconds(report.unloadSeconds))s\n")
        CLIStderr.write("  MP4 write: \(formatLTXSeconds(report.mp4WriteSeconds))s\n")
        CLIStderr.write("  end to end: \(formatLTXSeconds(report.totalSeconds))s\n")
    }

    guard let outputPath else { return }
    let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(to: outputURL, options: .atomic)
}

private func formatLTXSeconds(_ seconds: Double) -> String {
    String(format: "%.3f", seconds)
}
