@testable import StudioKit
import MereRunContract
import XCTest

/// Pins the draft every template starts from.
///
/// The fixture was recorded from the hand-written `defaultDraft()` switch that
/// `CommandDefaults` replaced, and lists only the fields a template moves off `CommandDraft`'s
/// own initial value. Re-record with `./scripts/update-studio-argv-fixture.sh` when a starting
/// value is meant to change, and say so in the pull request.
final class CommandDefaultDraftTests: XCTestCase {
    func testEveryTemplateStartsFromTheRecordedDraft() throws {
        let recorded = CommandDefaultDraftFixture.render()
        if ProcessInfo.processInfo.environment["MERERUN_UPDATE_STUDIO_ARGV_FIXTURE"] == "1" {
            try recorded.write(to: CommandDefaultDraftFixture.url, atomically: true, encoding: .utf8)
            return
        }

        let expected = try String(contentsOf: CommandDefaultDraftFixture.url, encoding: .utf8)
        guard recorded != expected else { return }

        let recordedLines = Set(recorded.split(separator: "\n").map(String.init))
        let expectedLines = Set(expected.split(separator: "\n").map(String.init))
        XCTAssertEqual(
            expectedLines.subtracting(recordedLines).sorted(), [], "Starting values no longer set"
        )
        XCTAssertEqual(
            recordedLines.subtracting(expectedLines).sorted(), [], "Starting values newly set"
        )
    }

    /// Every `contract` entry must name an option the contract really declares a default for,
    /// so an entry cannot quietly become a no-op when the contract drops a `default_value`.
    func testEveryContractDefaultResolves() throws {
        for (id, defaults) in CommandDefaults.byTemplate {
            let capability = id.capability
            for flag in defaults.compactMap(\.contractFlag) {
                let option = capability?.options.first { $0.flag == flag }
                XCTAssertNotNil(
                    option?.defaultValue,
                    "\(id) reads \(flag) from the contract, which declares no default for it"
                )
            }
        }
    }
}

enum CommandDefaultDraftFixture {
    static let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/command-default-drafts.txt")

    /// One `template<TAB>field<TAB>value` record per field a template moves off the shared
    /// `CommandDraft` initial value.
    static func render() -> String {
        let initial = Array(Mirror(reflecting: CommandDraft()).children)
        var lines: [String] = []
        for template in CommandCatalog.templates {
            let draft = Array(Mirror(reflecting: template.defaultDraft()).children)
            for (base, started) in zip(initial, draft) {
                let before = String(describing: base.value)
                let after = String(describing: started.value)
                guard before != after else { continue }
                lines.append("\(template.id.rawValue)\t\(started.label ?? "?")\t\(normalized(after))")
            }
        }
        return lines.sorted().joined(separator: "\n") + "\n"
    }

    /// Replaces the two parts of a starting value that differ between machines and runs: the
    /// home directory an output path sits under, and the timestamp in a default file name.
    private static func normalized(_ value: String) -> String {
        var value = value
        let home = NSHomeDirectory()
        if let range = value.range(of: home) {
            value.replaceSubrange(range, with: "~")
        }
        return value.replacing(/[0-9]{8}-[0-9]{6}/, with: "<stamp>")
    }
}
