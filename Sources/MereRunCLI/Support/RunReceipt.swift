import ArgumentParser
import Foundation

/// The structured result line long-running generation commands print as the
/// final stdout line when `--receipt` is set:
///
///     {"event":"result","exit":0,"outputs":[{"kind":"image","path":"/abs/out.png"}]}
///
/// The first output is always the primary artifact. Sidecars follow it and
/// carry a `role` (`detections`, `masks`, `recipe`, `lyrics`, ...). A receipt
/// is only printed after the command succeeded, so `exit` is always `0`; a
/// failed run exits nonzero without a receipt. Human output above the receipt
/// is unchanged, so wrappers parse the last line and ignore the rest.
struct RunReceipt: Codable, Equatable {
    enum OutputKind: String, Codable {
        case image
        case video
        case audio
        case text
        case json
        case directory
    }

    struct Output: Codable, Equatable {
        let path: String
        let kind: OutputKind
        let role: String?

        init(path: String, kind: OutputKind, role: String? = nil) {
            self.path = path
            self.kind = kind
            self.role = role
        }

        init(url: URL, kind: OutputKind, role: String? = nil) {
            self.init(path: url.standardizedFileURL.path, kind: kind, role: role)
        }
    }

    static let eventName = "result"
    static let flagName = "receipt"
    static let flagHelpText = "Print a final JSON result line to stdout listing every output path and kind."
    static let flagHelp = ArgumentHelp(flagHelpText)

    let event: String
    let outputs: [Output]
    let exit: Int32

    init(outputs: [Output], exit: Int32 = 0) {
        self.event = Self.eventName
        self.outputs = outputs
        self.exit = exit
    }

    /// One JSON object with no embedded newlines, suitable for NDJSON streams.
    func line() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(self)
        guard let line = String(data: data, encoding: .utf8) else {
            throw RunReceiptError.encoding
        }
        return line
    }

    /// Prints the receipt when `enabled`; a no-op otherwise so call sites can
    /// pass the flag through without branching.
    static func emit(
        _ outputs: [Output],
        enabled: Bool,
        write: (String) -> Void = { Swift.print($0) }
    ) throws {
        guard enabled else { return }
        write(try RunReceipt(outputs: outputs).line())
    }
}

/// Output lists for each command family, kept together so the primary-first
/// ordering and the sidecar roles stay consistent across commands.
extension RunReceipt {
    /// `image generate`: the PNG plus the optional structured-prompt JSON.
    static func generatedImageOutputs(image: URL, structuredPrompt: URL?) -> [Output] {
        var outputs = [Output(url: image, kind: .image)]
        if let structuredPrompt {
            outputs.append(Output(url: structuredPrompt, kind: .json, role: "structured-prompt"))
        }
        return outputs
    }

    /// `video generate`: the MP4 (or EXR directory) plus the optional timings JSON.
    static func generatedVideoOutputs(primary: URL, kind: OutputKind, timings: URL?) -> [Output] {
        var outputs = [Output(url: primary, kind: kind)]
        if let timings {
            outputs.append(Output(url: timings, kind: .json, role: "timings"))
        }
        return outputs
    }

    /// `music generate`, `sfx generate`, `speech synthesize`: one WAV plus any
    /// sidecars the lane wrote (candidates, stems, lyrics, recipe, bundle).
    static func generatedAudioOutputs(audio: URL, sidecars: [Output] = []) -> [Output] {
        [Output(url: audio, kind: .audio)] + sidecars
    }

    /// `speech transcribe`: the transcript file, or nothing when it only went to stdout.
    static func transcriptOutputs(_ transcript: URL?) -> [Output] {
        transcript.map { [Output(url: $0, kind: .text)] } ?? []
    }

    /// `vision ground` and `vision segment`: annotated image, detections JSON, optional mask directory.
    static func annotatedImageOutputs(image: URL, detections: URL, masks: URL?) -> [Output] {
        var outputs = [
            Output(url: image, kind: .image),
            Output(url: detections, kind: .json, role: "detections"),
        ]
        if let masks {
            outputs.append(Output(url: masks, kind: .directory, role: "masks"))
        }
        return outputs
    }

    /// `vision track`: annotated video, optional tracking JSON, optional mask directory.
    static func annotatedVideoOutputs(videoPath: String, trackingPath: String?, masks: URL?) -> [Output] {
        var outputs = [Output(path: videoPath, kind: .video)]
        if let trackingPath {
            outputs.append(Output(path: trackingPath, kind: .json, role: "tracking"))
        }
        if let masks {
            outputs.append(Output(url: masks, kind: .directory, role: "masks"))
        }
        return outputs
    }
}

enum RunReceiptError: Error, CustomStringConvertible {
    case encoding

    var description: String {
        switch self {
        case .encoding:
            return "Could not encode the run receipt as UTF-8."
        }
    }
}
