import ArgumentParser
import Foundation

struct ModelBenchmarkFusedFixture: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fused-fixture",
        abstract: "Stamp or verify normalized fused-benchmark JSONL fixture hashes."
    )

    @Argument(help: "Normalized JSON or JSONL fixture file.")
    var input: String

    @Flag(name: [.long], help: "Verify existing hashes without rewriting records to stdout.")
    var check: Bool = false

    func run() throws {
        let url = URL(fileURLWithPath: input).standardizedFileURL
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.split(whereSeparator: { $0.isNewline })
        guard !lines.isEmpty else {
            throw ValidationError("Fixture file is empty.")
        }
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

        for (index, line) in lines.enumerated() {
            let fixture: FusedExternalBenchmarkCase
            do {
                fixture = try decoder.decode(
                    FusedExternalBenchmarkCase.self,
                    from: Data(line.utf8)
                )
            } catch {
                throw ValidationError(
                    "\(url.path):\(index + 1): \(error.localizedDescription)"
                )
            }
            let stamped = try fixture.stamped()
            if check {
                guard fixture.contentSHA256 == stamped.contentSHA256 else {
                    throw ValidationError(
                        "\(fixture.id): content SHA-256 does not match."
                    )
                }
            } else {
                print(String(decoding: try encoder.encode(stamped), as: UTF8.self))
            }
        }
        if check {
            CLIStderr.write("Verified \(lines.count) fused benchmark fixture(s).\n")
        }
    }
}
