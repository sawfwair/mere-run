import ArgumentParser
import Foundation
import MereRunCore

struct VisionFlow: ParsableCommand {
    enum Accuracy: String, CaseIterable, ExpressibleByArgument {
        case low
        case medium
        case high
        case veryHigh = "very-high"

        var native: NativeOpticalFlowAccuracy {
            switch self {
            case .low: .low
            case .medium: .medium
            case .high: .high
            case .veryHigh: .veryHigh
            }
        }
    }

    static let configuration = CommandConfiguration(
        commandName: "flow",
        abstract: "Generate dense optical flow between two equal-size images."
    )

    @Argument(help: "Source image path.")
    var from: String

    @Argument(help: "Target image path.")
    var to: String

    @Option(name: [.customShort("o"), .long], help: "Middlebury .flo output path.")
    var output: String?

    @Option(name: [.customLong("json-output")], help: "JSON metadata output path.")
    var jsonOutput: String?

    @Option(name: [.long], help: "Flow accuracy: low, medium, high, or very-high. (default: high)")
    var accuracy: Accuracy = .high

    @Flag(name: [.long], help: "Print the metadata payload as JSON on stdout.")
    var json = false

    func run() throws {
        let fromURL = URL(fileURLWithPath: from).standardizedFileURL
        let toURL = URL(fileURLWithPath: to).standardizedFileURL
        let outputURL = Self.resolveFlowOutputURL(output, fromURL: fromURL, toURL: toURL)
        let jsonURL = Self.resolveJSONOutputURL(jsonOutput, flowOutputURL: outputURL)
        let result = try NativeOpticalFlowGenerator().generate(
            from: fromURL,
            to: toURL,
            outputURL: outputURL,
            accuracy: accuracy.native
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(result)
        try data.write(to: jsonURL, options: .atomic)
        if json {
            print(String(decoding: data, as: UTF8.self))
        } else {
            print(outputURL.path)
        }
    }

    static func resolveFlowOutputURL(_ raw: String?, fromURL: URL, toURL: URL) -> URL {
        if let raw, !raw.isEmpty {
            let url = URL(fileURLWithPath: raw).standardizedFileURL
            return url.pathExtension.isEmpty ? url.appendingPathExtension("flo") : url
        }
        let fromStem = fromURL.deletingPathExtension().lastPathComponent
        let toStem = toURL.deletingPathExtension().lastPathComponent
        return fromURL.deletingLastPathComponent().appendingPathComponent("\(fromStem)_to_\(toStem)_flow.flo")
    }

    static func resolveJSONOutputURL(_ raw: String?, flowOutputURL: URL) -> URL {
        if let raw, !raw.isEmpty {
            let url = URL(fileURLWithPath: raw).standardizedFileURL
            return url.pathExtension.isEmpty ? url.appendingPathExtension("json") : url
        }
        return flowOutputURL.deletingPathExtension().appendingPathExtension("json")
    }
}
