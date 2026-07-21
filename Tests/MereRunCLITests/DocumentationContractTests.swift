import ArgumentParser
import Foundation
import XCTest
@testable import MereRunCLI

final class DocumentationContractTests: XCTestCase {
    private struct CommandNode {
        let name: String
        let abstract: String
        let children: [CommandNode]
    }

    private struct CommandPage {
        let command: String
        let path: String
    }

    private static let topLevelStart = "<!-- BEGIN GENERATED: CLI TOP LEVEL -->"
    private static let topLevelEnd = "<!-- END GENERATED: CLI TOP LEVEL -->"
    private static let treeStart = "<!-- BEGIN GENERATED: CLI TREE -->"
    private static let treeEnd = "<!-- END GENERATED: CLI TREE -->"

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testGeneratedCommandInventoriesMatchCLI() throws {
        let commands = commandTree()
        let pages = try commandPages()
        let topLevel = renderTopLevel(commands: commands, pages: pages)
        let tree = renderTree(commands: commands, pages: pages)
        let indexURL = repositoryRoot.appendingPathComponent("docs/index.md")
        let gettingStartedURL = repositoryRoot.appendingPathComponent("docs/getting-started.md")
        let cliURL = repositoryRoot.appendingPathComponent("docs/cli.md")

        if ProcessInfo.processInfo.environment["MERERUN_UPDATE_DOCS"] == "1" {
            try replaceGeneratedBlock(
                in: indexURL,
                start: Self.topLevelStart,
                end: Self.topLevelEnd,
                with: topLevel
            )
            try replaceGeneratedBlock(
                in: gettingStartedURL,
                start: Self.topLevelStart,
                end: Self.topLevelEnd,
                with: topLevel
            )
            try replaceGeneratedBlock(
                in: cliURL,
                start: Self.treeStart,
                end: Self.treeEnd,
                with: tree
            )
        }

        let updateMessage = "Run ./scripts/update-docs-command-reference.sh and commit the generated changes."
        XCTAssertEqual(
            try generatedBlock(in: indexURL, start: Self.topLevelStart, end: Self.topLevelEnd),
            topLevel,
            updateMessage
        )
        XCTAssertEqual(
            try generatedBlock(in: gettingStartedURL, start: Self.topLevelStart, end: Self.topLevelEnd),
            topLevel,
            updateMessage
        )
        XCTAssertEqual(
            try generatedBlock(in: cliURL, start: Self.treeStart, end: Self.treeEnd),
            tree,
            updateMessage
        )
    }

    func testEveryPublicCommandHasNavigableDocumentation() throws {
        let commands = commandTree()
        let pages = try commandPages()
        XCTAssertEqual(
            pages.map(\.command),
            commands.map(\.name),
            "docs/.vitepress/command-pages.tsv must own every top-level CLI command in CLI order."
        )

        let navigation = try String(
            contentsOf: repositoryRoot.appendingPathComponent("docs/.vitepress/config.mts"),
            encoding: .utf8
        )
        for page in pages {
            let basePath = page.path.split(separator: "#", maxSplits: 1).first.map(String.init) ?? page.path
            let relativePath = basePath == "/" ? "index.md" : "\(basePath.dropFirst()).md"
            let pageURL = repositoryRoot.appendingPathComponent("docs/\(relativePath)")
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: pageURL.path),
                "Documentation owner for mere.run \(page.command) does not exist: \(relativePath)"
            )
            XCTAssertTrue(
                navigation.contains("link: '\(basePath)'"),
                "The documentation owner for mere.run \(page.command) is missing from the sidebar: \(basePath)"
            )
        }

        let runtimeDirectory = repositoryRoot.appendingPathComponent("docs/runtime")
        let runtimePages = try FileManager.default.contentsOfDirectory(
            at: runtimeDirectory,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "md" }
        for pageURL in runtimePages {
            let path = "/runtime/\(pageURL.deletingPathExtension().lastPathComponent)"
            XCTAssertTrue(navigation.contains("link: '\(path)'"), "Runtime page is missing from the sidebar: \(path)")
        }
    }

    func testDocumentedCommandInvocationsUseCurrentCommandTree() throws {
        let commands = commandTree()
        let docsRoot = repositoryRoot.appendingPathComponent("docs")
        let markdownFiles = try FileManager.default.subpathsOfDirectory(atPath: docsRoot.path)
            .filter { $0.hasSuffix(".md") }
            .sorted()
        var failures: [String] = []

        for relativePath in markdownFiles {
            let fileURL = docsRoot.appendingPathComponent(relativePath)
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            var inFence = false
            for (offset, lineValue) in contents.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let line = String(lineValue)
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("```") {
                    inFence.toggle()
                    continue
                }

                let candidates = inlineCommandCandidates(in: line) + (inFence ? fencedCommandCandidates(in: line) : [])
                for candidate in candidates {
                    if let invalid = invalidCommandPath(candidate, commands: commands) {
                        failures.append("\(relativePath):\(offset + 1): \(invalid)")
                    }
                }
            }
        }

        XCTAssertTrue(
            failures.isEmpty,
            "Documented command paths must match the CLI command tree:\n\(failures.joined(separator: "\n"))"
        )
    }

    func testGraphStudioDocumentationPreservesTheVersionBoundary() throws {
        let studioURL = repositoryRoot.appendingPathComponent("docs/graph/studio.md")
        let navigationURL = repositoryRoot.appendingPathComponent("docs/.vitepress/config.mts")
        let workflowsURL = repositoryRoot.appendingPathComponent("docs/workflows.md")
        let studio = try String(contentsOf: studioURL, encoding: .utf8)
        let navigation = try String(contentsOf: navigationURL, encoding: .utf8)
        let workflows = try String(contentsOf: workflowsURL, encoding: .utf8)

        XCTAssertTrue(navigation.contains("link: '/graph/studio'"))
        XCTAssertTrue(studio.contains("Graph v2"))
        XCTAssertTrue(studio.contains("`schema_version: 1`"))
        XCTAssertTrue(studio.contains("https://studio.mere.run/"))
        XCTAssertTrue(workflows.contains("Graph v2 runtime and v1 contract"))
        XCTAssertTrue(studio.contains("Do not change a workflow to `schema_version: 2`"))
        XCTAssertTrue(studio.contains("Local Studio never requires Mere World sign-in"))
        XCTAssertTrue(studio.contains("Mere World OAuth Authorization Code with PKCE"))
        XCTAssertTrue(studio.contains("OAuth device grant"))
        XCTAssertTrue(studio.contains("does not open a browser callback or local web server"))
        XCTAssertTrue(studio.contains("does not ship a Python"))
    }

    private func commandTree() -> [CommandNode] {
        nodes(from: MereRunCLI.configuration.subcommands)
    }

    private func nodes(from commandTypes: [ParsableCommand.Type]) -> [CommandNode] {
        commandTypes.compactMap { commandType in
            let configuration = commandType.configuration
            guard configuration.shouldDisplay, let name = configuration.commandName else { return nil }
            return CommandNode(
                name: name,
                abstract: normalized(configuration.abstract),
                children: nodes(from: configuration.subcommands)
            )
        }
    }

    private func normalized(_ value: String) -> String {
        value.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    private func commandPages() throws -> [CommandPage] {
        let mapURL = repositoryRoot.appendingPathComponent("docs/.vitepress/command-pages.tsv")
        let contents = try String(contentsOf: mapURL, encoding: .utf8)
        return try contents.split(separator: "\n").compactMap { lineValue in
            let line = String(lineValue)
            guard !line.isEmpty, !line.hasPrefix("#") else { return nil }
            let columns = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard columns.count == 2, !columns[0].isEmpty, columns[1].hasPrefix("/") else {
                throw NSError(
                    domain: "DocumentationContractTests",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Invalid command-pages.tsv row: \(line)"]
                )
            }
            return CommandPage(command: String(columns[0]), path: String(columns[1]))
        }
    }

    private func renderTopLevel(commands: [CommandNode], pages: [CommandPage]) -> String {
        let paths = Dictionary(uniqueKeysWithValues: pages.map { ($0.command, $0.path) })
        var lines = ["| Command | Purpose |", "| --- | --- |"]
        for command in commands {
            let path = paths[command.name] ?? "/cli"
            let abstract = command.abstract.replacingOccurrences(of: "|", with: "\\|")
            lines.append("| [`mere.run \(command.name)`](\(path)) | \(abstract) |")
        }
        return lines.joined(separator: "\n")
    }

    private func renderTree(commands: [CommandNode], pages: [CommandPage]) -> String {
        let paths = Dictionary(uniqueKeysWithValues: pages.map { ($0.command, $0.path) })
        var lines: [String] = []
        for command in commands {
            let path = paths[command.name] ?? "/cli"
            lines.append("- [`mere.run \(command.name)`](\(path)) — \(command.abstract)")
            appendChildren(command.children, prefix: [command.name], depth: 1, to: &lines)
        }
        return lines.joined(separator: "\n")
    }

    private func appendChildren(
        _ commands: [CommandNode],
        prefix: [String],
        depth: Int,
        to lines: inout [String]
    ) {
        for command in commands {
            let path = prefix + [command.name]
            lines.append("\(String(repeating: "  ", count: depth))- `mere.run \(path.joined(separator: " "))` — \(command.abstract)")
            appendChildren(command.children, prefix: path, depth: depth + 1, to: &lines)
        }
    }

    private func generatedBlock(in url: URL, start: String, end: String) throws -> String {
        let contents = try String(contentsOf: url, encoding: .utf8)
        let startToken = "\(start)\n"
        let endToken = "\n\(end)"
        guard let startRange = contents.range(of: startToken),
              let endRange = contents.range(of: endToken, range: startRange.upperBound..<contents.endIndex) else {
            throw NSError(
                domain: "DocumentationContractTests",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Missing generated block markers in \(url.path)"]
            )
        }
        return String(contents[startRange.upperBound..<endRange.lowerBound])
    }

    private func replaceGeneratedBlock(in url: URL, start: String, end: String, with replacement: String) throws {
        var contents = try String(contentsOf: url, encoding: .utf8)
        let startToken = "\(start)\n"
        let endToken = "\n\(end)"
        guard let startRange = contents.range(of: startToken),
              let endRange = contents.range(of: endToken, range: startRange.upperBound..<contents.endIndex) else {
            throw NSError(
                domain: "DocumentationContractTests",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Missing generated block markers in \(url.path)"]
            )
        }
        contents.replaceSubrange(startRange.upperBound..<endRange.lowerBound, with: replacement)
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }

    private func inlineCommandCandidates(in line: String) -> [String] {
        let pattern = #"`((?:swift run )?mere\.run(?: [a-z][a-z0-9-]*)+)`"#
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(line.startIndex..<line.endIndex, in: line)
        return expression.matches(in: line, range: range).compactMap { match in
            guard let candidateRange = Range(match.range(at: 1), in: line) else { return nil }
            return String(line[candidateRange])
        }
    }

    private func fencedCommandCandidates(in line: String) -> [String] {
        var trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("$ ") || trimmed.hasPrefix("| ") {
            trimmed.removeFirst(2)
        }
        guard trimmed.hasPrefix("mere.run ") || trimmed.hasPrefix("swift run mere.run ") else { return [] }
        return [trimmed]
    }

    private func invalidCommandPath(_ candidate: String, commands: [CommandNode]) -> String? {
        let normalizedCandidate = candidate.hasPrefix("swift run ")
            ? String(candidate.dropFirst("swift run ".count))
            : candidate
        let tokens = normalizedCandidate.split(whereSeparator: \.isWhitespace).map(String.init)
        guard tokens.first == "mere.run", tokens.count > 1 else { return nil }
        var siblings = commands
        var path: [String] = []

        for token in tokens.dropFirst() {
            guard !token.hasPrefix("-"), token.range(of: #"^[a-z][a-z0-9-]*$"#, options: .regularExpression) != nil else {
                return nil
            }
            guard !siblings.isEmpty else { return nil }
            guard let command = siblings.first(where: { $0.name == token }) else {
                let parent = path.isEmpty ? "mere.run" : "mere.run \(path.joined(separator: " "))"
                return "`\(candidate)` uses unknown subcommand `\(token)` under `\(parent)`"
            }
            path.append(token)
            siblings = command.children
        }
        return nil
    }
}
