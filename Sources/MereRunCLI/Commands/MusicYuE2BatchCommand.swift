import ArgumentParser
import AudioCore
import Foundation
import MereRunCore

struct YuE2Export: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "export", abstract: "Copy a verified native YuE2 pipeline for offline use.")
    @OptionGroup var options: YuE2StageOptions
    func run() throws {
        let root = options.model == "music-yue2" ? try ModelResolver().resolve(.yue2).rootURL : YuE2Transcribe.url(options.model)
        let receipt = try YuE2Resources(rootURL: root).export(to: YuE2Transcribe.url(options.output))
        print(String(decoding: try JSONEncoder().encode(receipt), as: UTF8.self))
    }
}

struct YuE2Batch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "batch", abstract: "Run independent YuE2 JSONL requests and retain per-request failures.")
    struct Item: Decodable { let id: String; let request: YuE2GenerationPlan }
    struct Outcome: Encodable { let line: Int; let id: String?; let output: String?; let error: String? }
    @Argument(help: "JSONL file: each line contains id and request.") var input: String
    @OptionGroup var options: YuE2StageOptions
    @Flag(name: .customLong("validate-only"), help: "Validate IDs and requests without loading models or writing outputs.") var validateOnly = false
    func run() async throws {
        let text = try String(contentsOf: YuE2Transcribe.url(input), encoding: .utf8)
        let destination = YuE2Transcribe.url(options.output)
        let lines = text.components(separatedBy: .newlines)
        var seen = Set<String>()
        var operation: YuE2GenerationOperation?
        for (index, line) in lines.enumerated() where !line.trimmingCharacters(in: .whitespaces).isEmpty {
            try Task.checkCancellation()
            var id: String?
            let outcome: Outcome
            do {
                let item = try JSONDecoder().decode(Item.self, from: Data(line.utf8))
                id = item.id
                guard item.id.range(of: #"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$"#, options: .regularExpression) != nil,
                      seen.insert(item.id).inserted else { throw ValidationError("Invalid or duplicate request ID.") }
                let output = destination.appendingPathComponent(item.id)
                guard !FileManager.default.fileExists(atPath: output.path) else { throw ValidationError("Request output already exists.") }
                if !validateOnly {
                    if operation == nil { operation = try options.operation() }
                    guard let operation else { throw ValidationError("Unable to initialize YuE2.") }
                    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
                    let generated = try await operation.generate(item.request, stageOutput: output, progress: YuE2StageOptions.progress)
                    _ = try AudioExportService.write(generated.waveform, plan: AudioExportPlan(options: .music),
                        to: output.appendingPathComponent("audio.wav"))
                }
                outcome = Outcome(line: index + 1, id: id, output: validateOnly ? nil : output.path, error: nil)
            } catch is CancellationError { throw CancellationError() }
            catch { outcome = Outcome(line: index + 1, id: id, output: nil, error: error.localizedDescription) }
            print(String(decoding: try JSONEncoder().encode(outcome), as: UTF8.self))
        }
    }
}
