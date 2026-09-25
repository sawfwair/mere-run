import Foundation
import MereRunContract
import MereRunCore
import XCTest

@testable import MereRunCLI

/// Nothing that ran on main may start failing at the capability gate. `main-behaviour.json`
/// freezes what main did with every command line the reviewers ran against it (and main's own
/// tests and documented invocations, and the spellings the final review named): `passes` when it
/// got past every option and model check, `refuses` when it stopped on the command line itself.
/// The gate may refuse only what main refused; a row where the branch knowingly differs says why.
///
/// XCTest runs these one at a time, so the process environment and the model store can be
/// pinned to what the table was recorded with: an empty store and no `MERERUN_*` settings.
final class MainBehaviourRegressionTests: XCTestCase {
    private struct Table: Decodable {
        let stores: [String: Store]
        let rows: [Row]
    }

    /// A model folder some rows name through `$FIXTURE/<name>`.
    private struct Store: Decodable {
        let files: [String: String]
        let directories: [String]
    }

    private struct Row: Decodable {
        enum Outcome: String, Decodable {
            case passes
            case refuses
        }

        let argv: [String]
        let main: Outcome
        let source: String
        let evidence: String
        let env: [String: String]?
        /// Managed ids installed for this row, each as a copy of the named store.
        let installed: [String: String]?
        let acceptedDifference: String?

        enum CodingKeys: String, CodingKey {
            case argv, main, source, evidence, env, installed
            case acceptedDifference = "accepted_difference"
        }
    }

    private var savedEnvironment: [String: String] = [:]
    private var root = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedEnvironment = ProcessInfo.processInfo.environment.filter { $0.key.hasPrefix("MERERUN_") }
        for key in savedEnvironment.keys {
            unsetenv(key)
        }
        root = FileManager.default.temporaryDirectory.appendingPathComponent("main-behaviour-\(UUID().uuidString)")
        let models = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        MereRunModelPaths.setProcessModelsDirOverride(models)
    }

    override func tearDown() {
        MereRunModelPaths.setProcessModelsDirOverride(nil)
        for (key, value) in savedEnvironment {
            setenv(key, value, 1)
        }
        try? FileManager.default.removeItem(at: root)
        super.tearDown()
    }

    private func table() throws -> Table {
        let resources = try XCTUnwrap(Bundle.module.resourceURL)
        let url = resources.appendingPathComponent("Fixtures/MainBehaviour/main-behaviour.json")
        return try JSONDecoder().decode(Table.self, from: Data(contentsOf: url))
    }

    /// Builds each store the rows name, as the recording built it.
    private func stores(_ table: Table) throws -> URL {
        let stores = root.appendingPathComponent("stores", isDirectory: true)
        for (name, store) in table.stores {
            let folder = stores.appendingPathComponent(name, isDirectory: true)
            for directory in store.directories {
                try FileManager.default.createDirectory(
                    at: folder.appendingPathComponent(directory, isDirectory: true), withIntermediateDirectories: true
                )
            }
            for (path, contents) in store.files {
                let file = folder.appendingPathComponent(path)
                try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data(contents.utf8).write(to: file)
            }
        }
        return stores
    }

    /// Installs each managed id as a copy of its store, with the manifest a pull writes.
    private func install(_ installed: [String: String], from stores: String) throws -> [URL] {
        try installed.map { id, store in
            let model = try XCTUnwrap(ModelResolver.ModelID(rawValue: id), id)
            let folder = MereRunModelPaths.modelsDir.appendingPathComponent(id, isDirectory: true)
            try FileManager.default.copyItem(at: URL(fileURLWithPath: stores).appendingPathComponent(store), to: folder)
            try MereRunModelManifest.template(for: model, createdAt: Date(timeIntervalSince1970: 0)).write(to: folder)
            return folder
        }
    }

    func testTheGateNeverRefusesACommandLineMainRan() throws {
        let table = try table()
        let stores = try stores(table).path
        var checked = 0
        for row in table.rows where row.main == .passes && row.acceptedDifference == nil {
            let environment = (row.env ?? [:]).mapValues { $0.replacingOccurrences(of: "$FIXTURE", with: stores) }
            for (key, value) in environment {
                setenv(key, value, 1)
            }
            defer {
                for key in environment.keys {
                    unsetenv(key)
                }
            }
            let installed = try install(row.installed ?? [:], from: stores)
            defer {
                for folder in installed {
                    try? FileManager.default.removeItem(at: folder)
                }
            }
            let argv = row.argv.map { $0.replacingOccurrences(of: "$FIXTURE", with: stores) }
            XCTAssertNoThrow(
                try CLICapabilityGate.check(arguments: ["mere.run"] + argv),
                "[\(row.source)] mere.run \(argv.joined(separator: " ")) ran on main (\(row.evidence))"
            )
            checked += 1
        }
        XCTAssertGreaterThan(checked, 800, "the table covers the reviewers' harnesses")
    }

    /// The table carries every spelling and environment the final review named, and says why each
    /// knowing difference from main is one.
    func testTheTableCoversEverySpellingTheReviewNamed() throws {
        let table = try table()
        let kinds = Set(table.rows.compactMap { row -> String? in
            let parts = row.source.split(separator: ":")
            return parts.first == "spelling" ? String(parts[1]) : nil
        })
        XCTAssertEqual(
            kinds,
            ["repeated", "grouped", "equals", "empty", "case", "builtin", "listing", "model", "root", "env", "folder", "finding"]
        )
        XCTAssertTrue(table.rows.contains { $0.env?.isEmpty == false })
        for row in table.rows where row.acceptedDifference != nil {
            XCTAssertEqual(row.main, .passes, "\(row.source): only a row main passes needs a reason to differ")
            XCTAssertFalse(row.acceptedDifference?.isEmpty ?? true, "\(row.source)")
        }
    }
}
