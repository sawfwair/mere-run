import ArgumentParser
import Foundation
import MereRunCore

struct VisionPose: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pose",
        abstract: "Detect body, hand, and face landmarks in an image with the native platform runtime."
    )

    @Argument(help: "Image file path.")
    var image: String

    @Option(name: [.customLong("json-output")], help: "JSON landmark output path (default: <image>_pose.json).")
    var jsonOutput: String?

    @Flag(name: [.customLong("no-body")], help: "Skip human body landmarks.")
    var noBody = false

    @Flag(name: [.customLong("no-hands")], help: "Skip hand landmarks.")
    var noHands = false

    @Flag(name: [.customLong("no-face")], help: "Skip face landmarks.")
    var noFace = false

    @Option(name: [.customLong("max-hands")], help: "Maximum hands to detect. (default: 2)")
    var maxHands = 2

    @Option(name: [.customLong("minimum-confidence")], help: "Minimum landmark confidence in [0, 1]. (default: 0.1)")
    var minimumConfidence = 0.1

    @Flag(name: [.long], help: "Print the landmark payload as JSON on stdout.")
    var json = false

    func validate() throws {
        guard !noBody || !noHands || !noFace else {
            throw ValidationError("At least one of body, hands, or face must remain enabled.")
        }
        guard maxHands > 0 else {
            throw ValidationError("--max-hands must be greater than zero.")
        }
        guard (0.0...1.0).contains(minimumConfidence) else {
            throw ValidationError("--minimum-confidence must be between 0 and 1.")
        }
    }

    func run() throws {
        let imageURL = URL(fileURLWithPath: image).standardizedFileURL
        let outputURL = Self.resolveJSONOutputURL(jsonOutput, inputImageURL: imageURL)
        let request = NativePoseRequest(
            includeBody: !noBody,
            includeHands: !noHands,
            includeFace: !noFace,
            maximumHandCount: maxHands,
            minimumConfidence: Float(minimumConfidence)
        )
        let result = try NativePoseDetector().detect(imageURL: imageURL, request: request)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(result)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: outputURL, options: .atomic)
        if json {
            print(String(decoding: data, as: UTF8.self))
        } else {
            print(outputURL.path)
        }
    }

    static func resolveJSONOutputURL(_ rawOutput: String?, inputImageURL: URL) -> URL {
        guard let rawOutput, !rawOutput.isEmpty else {
            let directory = inputImageURL.deletingLastPathComponent()
            let stem = inputImageURL.deletingPathExtension().lastPathComponent
            return directory.appendingPathComponent("\(stem)_pose.json")
        }
        let outputURL = URL(fileURLWithPath: rawOutput).standardizedFileURL
        return outputURL.pathExtension.isEmpty ? outputURL.appendingPathExtension("json") : outputURL
    }
}
