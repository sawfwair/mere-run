@testable import MereRunApp
import XCTest

/// Pins the argv every template builds, so a refactor of the argument builder cannot change
/// what the app runs.
///
/// The fixture was recorded from the hand-written `arguments(from:)` switch that
/// `ArgumentBuilder` replaced. Each record is one template built from one representative
/// draft: the template's own defaults, a draft with every field moved off its default with
/// booleans on and off, and one draft per hand-named variant that selects a model family or
/// a task mode. Re-record with `./scripts/update-studio-argv-fixture.sh` only when a change
/// to the command line is intended, and say so in the pull request.
final class CommandArgumentGoldenTests: XCTestCase {
    func testEveryTemplateBuildsTheRecordedArgv() throws {
        let recorded = try CommandArgumentFixture.render()
        if ProcessInfo.processInfo.environment["MERERUN_UPDATE_STUDIO_ARGV_FIXTURE"] == "1" {
            try recorded.write(to: CommandArgumentFixture.url, atomically: true, encoding: .utf8)
            return
        }

        let expected = try String(contentsOf: CommandArgumentFixture.url, encoding: .utf8)
        guard recorded != expected else { return }

        let recordedLines = recorded.split(separator: "\n", omittingEmptySubsequences: false)
        let expectedLines = expected.split(separator: "\n", omittingEmptySubsequences: false)
        for (index, expectedLine) in expectedLines.enumerated() {
            let recordedLine = index < recordedLines.count ? recordedLines[index] : "<missing>"
            XCTAssertEqual(
                String(recordedLine),
                String(expectedLine),
                """
                Template argv changed. If the new command line is intended, re-record with \
                ./scripts/update-studio-argv-fixture.sh and explain the change in the pull request.
                """
            )
            if recordedLine != expectedLine { return }
        }
        XCTFail("Recorded \(recordedLines.count) argv lines but the fixture holds \(expectedLines.count)")
    }
}

enum CommandArgumentFixture {
    static let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/command-argv.txt")

    /// One `template<TAB>draft<TAB>argv` record per line, sorted so the catalog's own order
    /// cannot move a record, and argv shell-quoted so a record stays on one line and a
    /// whitespace change inside an argument is still visible.
    static func render() throws -> String {
        var lines: [String] = []
        for template in CommandCatalog.templates {
            for draft in try drafts(for: template) {
                let argv = template.arguments(from: draft.draft)
                    .map(normalized)
                    .map { $0.contains(where: \.isWhitespace) || $0.isEmpty ? "'\($0)'" : $0 }
                    .joined(separator: " ")
                lines.append("\(template.id.rawValue)\t\(draft.name)\t\(argv)")
            }
        }
        return lines.sorted().joined(separator: "\n") + "\n"
    }

    /// Replaces the two parts of an argument that differ between machines and runs: the home
    /// directory an output path sits under, and the timestamp `StudioOutputLocation` stamps
    /// into a default file name.
    private static func normalized(_ argument: String) -> String {
        var argument = argument
        let home = NSHomeDirectory()
        if argument.hasPrefix(home) {
            argument = "~" + argument.dropFirst(home.count)
        }
        return argument.replacing(/[0-9]{8}-[0-9]{6}/, with: "<stamp>")
    }

    /// The recorded drafts: the template's own defaults, every field moved off its default
    /// with booleans on and then off, and one draft per hand-named and per enum-case variant.
    ///
    /// `CommandDraftProbes.probes(for:)` runs each variant twice more, once per boolean
    /// polarity. The fixture is a committed file, so it records each variant once and leaves
    /// the doubled sweep to `CommandContractGuardTests`, which builds it in memory.
    private static func drafts(for template: CommandTemplate) throws -> [CommandDraftProbes.Probe] {
        let defaults = template.defaultDraft()
        var drafts = [
            CommandDraftProbes.Probe(name: "default", draft: defaults),
            CommandDraftProbes.Probe(
                name: "maximal", draft: try CommandDraftProbes.maximalDraft(from: defaults, booleans: true)
            ),
            CommandDraftProbes.Probe(
                name: "maximal-booleans-off",
                draft: try CommandDraftProbes.maximalDraft(from: defaults, booleans: false)
            )
        ]
        var variants = CommandDraftProbes.variantValuesByField.map { ($0.key, $0.value.map { $0 as Any }) }
        variants += CommandDraftProbes.enumCaseVariants(in: defaults)
        for (field, values) in variants.sorted(by: { $0.0 < $1.0 }) {
            for value in values {
                drafts.append(CommandDraftProbes.Probe(
                    name: "\(field)=\(value)",
                    draft: try CommandDraftProbes.maximalDraft(
                        from: defaults, booleans: true, overriding: [field: value]
                    )
                ))
            }
        }
        return drafts
    }
}
