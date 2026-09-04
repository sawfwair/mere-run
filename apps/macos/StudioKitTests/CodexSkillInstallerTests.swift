@testable import StudioKit
import XCTest

final class CodexSkillInstallerTests: XCTestCase {
    func testInstallsBundledCodexSkills() throws {
        let rootURL = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: rootURL) }

        let sourceURL = rootURL.appendingPathComponent("skills", isDirectory: true)
        let destinationURL = rootURL.appendingPathComponent("codex-skills", isDirectory: true)
        try makeSkill(named: "use-mere-run", at: sourceURL)
        try FileManager.default.createDirectory(
            at: sourceURL.appendingPathComponent("not-a-skill", isDirectory: true),
            withIntermediateDirectories: true
        )

        let installed = try CodexSkillInstaller.installSkills(from: sourceURL, to: destinationURL)

        XCTAssertEqual(installed, ["use-mere-run"])
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: destinationURL
                    .appendingPathComponent("use-mere-run/agents/openai.yaml", isDirectory: false)
                    .path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: destinationURL
                    .appendingPathComponent("not-a-skill", isDirectory: true)
                    .path
            )
        )
    }

    func testDefaultCodexDestinationMatchesGlobalCodexSkillsFolder() {
        let destination = CodexSkillInstaller.defaultCodexSkillsDirectory()

        XCTAssertEqual(destination.lastPathComponent, "skills")
        XCTAssertEqual(destination.deletingLastPathComponent().lastPathComponent, ".codex")
    }

    func testMissingSkillManifestsThrowsDiagnosticError() throws {
        let rootURL = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: rootURL) }

        let sourceURL = rootURL.appendingPathComponent("skills", isDirectory: true)
        let destinationURL = rootURL.appendingPathComponent("codex-skills", isDirectory: true)
        try FileManager.default.createDirectory(at: sourceURL, withIntermediateDirectories: true)

        XCTAssertThrowsError(
            try CodexSkillInstaller.installSkills(from: sourceURL, to: destinationURL)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("No skill folders containing SKILL.md"))
        }
    }

    private func makeSkill(named name: String, at rootURL: URL) throws {
        let skillURL = rootURL.appendingPathComponent(name, isDirectory: true)
        let agentsURL = skillURL.appendingPathComponent("agents", isDirectory: true)
        try FileManager.default.createDirectory(at: agentsURL, withIntermediateDirectories: true)
        try "---\nname: \(name)\n---\n".write(
            to: skillURL.appendingPathComponent("SKILL.md", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )
        try "agent: codex\n".write(
            to: agentsURL.appendingPathComponent("openai.yaml", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )
    }

    private func temporaryDirectory() throws -> URL {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("CodexSkillInstallerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        return rootURL
    }
}
