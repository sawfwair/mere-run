import ArgumentParser
import Foundation
import MereRunContract
import MereRunRelayKit
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class SkillContractTests: XCTestCase {
    private struct Example {
        let location: String
        let arguments: [String]
        let block: Int
    }

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testSkillExamplesMatchCLIAndModelContracts() throws {
        for document in try skillDocuments() {
            let examples = try examples(in: document)
            var pulledModels = Set<String>()
            var preflights: [String: [String]] = [:]
            for example in examples {
                do {
                    // Parse the leaf directly: root validation bootstraps machine-local state.
                    // Never invoke run(), model resolution, or a shell for documentation checks.
                    let command = try parse(example.arguments)
                    if let pull = command as? ModelPull {
                        let model = try XCTUnwrap(pull.target, example.location)
                        try checkModel(model, location: example.location)
                        if !pull.preflight { pulledModels.insert(model) }
                    }
                    try checkDiscovery(command, location: example.location)
                    try checkRecipe(command, example: example, pulledModels: pulledModels)
                    if let command, try recipeModel(command, example: example) != nil {
                        let key = example.arguments.prefix(2).joined(separator: " ")
                        if example.arguments.contains("--preflight") {
                            preflights[key] = example.arguments
                        } else if type(of: command).helpMessage().contains("--preflight") {
                            let checked = try XCTUnwrap(preflights[key], "\(example.location): missing request preflight")
                            try matchingRequest(checked, execution: example.arguments)
                        }
                    }
                } catch {
                    XCTFail("\(example.location): \(error)")
                }
            }
        }
    }

    func testRecipeCheckRejectsRegressionsInDocumentedImageRequest() throws {
        let document = repositoryRoot.appendingPathComponent("skills/use-mere-run/references/preflight-and-actions.md")
        let examples = try examples(in: document)
        let example = try XCTUnwrap(examples.first {
            $0.arguments.starts(with: ["image", "generate"]) && !$0.arguments.contains("--preflight")
        })
        let checked = try XCTUnwrap(examples.first {
            $0.arguments.starts(with: ["image", "generate"]) && $0.arguments.contains("--preflight")
        })
        let modelIndex = try XCTUnwrap(example.arguments.firstIndex(of: "--model"))
        var implicit = example.arguments
        implicit.removeSubrange(modelIndex...modelIndex + 1)
        XCTAssertThrowsError(try recipeModel(parse(implicit), example: example))

        var mismatched = example.arguments
        mismatched[modelIndex + 1] = "image-klein-max"
        XCTAssertThrowsError(try checkRecipe(
            parse(mismatched), example: example, pulledModels: ["image-zimage-nano"]
        ))
        var changedSeed = example.arguments
        let seedIndex = try XCTUnwrap(changedSeed.firstIndex(of: "--seed"))
        changedSeed[seedIndex + 1] = "99"
        XCTAssertThrowsError(try matchingRequest(checked.arguments, execution: changedSeed))
        XCTAssertNoThrow(try matchingRequest(checked.arguments, execution: example.arguments))
    }

    func testBundledReferencesResolveAndExampleGraphValidates() throws {
        let root = repositoryRoot.appendingPathComponent("skills/use-mere-run")
        let skill = try String(contentsOf: root.appendingPathComponent("SKILL.md"), encoding: .utf8)
        let links = try NSRegularExpression(pattern: #"\]\((references/[^)]+)\)"#)
        let matches = links.matches(in: skill, range: NSRange(skill.startIndex..., in: skill))
        let linked = try Set(matches.map { match -> String in
            let range = try XCTUnwrap(Range(match.range(at: 1), in: skill))
            let relative = String(skill[range])
            XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent(relative).path))
            return relative
        })
        let referenceFiles = try FileManager.default.subpathsOfDirectory(atPath: root.appendingPathComponent("references").path)
            .filter { $0.hasSuffix(".md") }.map { "references/\($0)" }
        XCTAssertEqual(linked, Set(referenceFiles), "Every bundled operational reference must be discoverable")

        let workflow = try String(contentsOf: root.appendingPathComponent("references/workflows-and-services.md"), encoding: .utf8)
        let graphJSON = try XCTUnwrap(workflow.components(separatedBy: "```json\n").dropFirst().first?
            .components(separatedBy: "```").first)
        let graph = try WorkflowBundleCodec.decoder().decode(WorkflowGraphDocument.self, from: Data(graphJSON.utf8))
        let result = WorkflowGraphValidator.validate(graph: graph, inputs: .init(values: [:]))
        XCTAssertNotEqual(result.status, .blocked)
        XCTAssertEqual(result.order, ["message"])
    }

    func testDocumentedResultAndProgressJSONMatchEmitters() throws {
        let document = repositoryRoot.appendingPathComponent("skills/use-mere-run/references/execution-and-results.md")
        let markdown = try String(contentsOf: document, encoding: .utf8)
        let json = markdown.components(separatedBy: "```json\n").dropFirst().compactMap {
            $0.components(separatedBy: "```").first
        }
        XCTAssertEqual(json.count, 2)
        let receipt = try JSONDecoder().decode(RunReceipt.self, from: Data(try XCTUnwrap(json.first).utf8))
        XCTAssertEqual(receipt, RunReceipt(outputs: receipt.outputs))
        struct Progress: Decodable, Equatable {
            let event: String
            let stage: String
            let step: Int
            let total_steps: Int
        }
        let progress = try JSONDecoder().decode(Progress.self, from: Data(try XCTUnwrap(json.last).utf8))
        let emitted = CLIGenerationProgressPrinter.progressJSONLine(
            stage: progress.stage, step: progress.step, totalSteps: progress.total_steps
        )
        XCTAssertEqual(progress, try JSONDecoder().decode(Progress.self, from: Data(emitted.utf8)))
    }

    func testDocumentedPreflightsExposeBlockersWithoutInference() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("skill-preflight-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let recipes = try examples(in: repositoryRoot.appendingPathComponent(
            "skills/use-mere-run/references/preflight-and-actions.md"
        ))
        let serviceRecipes = try examples(in: repositoryRoot.appendingPathComponent(
            "skills/use-mere-run/references/workflows-and-services.md"
        ))
        struct Report: Decodable {
            let status: String
            let diagnostics: [Diagnostic]
            struct Diagnostic: Decodable { let id: String; let severity: String }
        }
        for (path, diagnostic, expectedExit) in [
            (["image", "generate"], "model_missing", Int32(1)),
            (["text", "chat"], "text_chat_model_not_installed", Int32(0)),
            (["api", "serve"], "api_key_required_for_non_loopback", Int32(1)),
        ] {
            let example = try XCTUnwrap((recipes + serviceRecipes).first {
                $0.arguments.starts(with: path) && $0.arguments.contains("--preflight")
            })
            var arguments = example.arguments
            if path == ["api", "serve"] {
                let host = try XCTUnwrap(arguments.firstIndex(of: "--host"))
                arguments[host + 1] = "0.0.0.0"
            }
            let output = try runCLI(arguments, in: root)
            let context = "Command: \(arguments.joined(separator: " "))\n"
                + "Exit: \(output.exit)\nStderr: \(String(decoding: output.stderr, as: UTF8.self))\n"
                + "Stdout: \(String(decoding: output.stdout, as: UTF8.self))"
            XCTAssertEqual(output.exit, expectedExit, context)
            let report: Report
            do {
                report = try JSONDecoder().decode(Report.self, from: output.stdout)
            } catch {
                XCTFail("Preflight did not emit its JSON report: \(error)\n\(context)")
                continue
            }
            XCTAssertEqual(report.status, "blocked")
            XCTAssertTrue(report.diagnostics.contains { $0.id == diagnostic && $0.severity == "blocker" })
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("mug.png").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("models").path))
    }

    func testDocumentedGraphRunsInspectsAndResumesWithoutModels() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("skill-graph-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let document = repositoryRoot.appendingPathComponent("skills/use-mere-run/references/workflows-and-services.md")
        let markdown = try String(contentsOf: document, encoding: .utf8)
        let json = try XCTUnwrap(markdown.components(separatedBy: "```json\n").dropFirst().first?
            .components(separatedBy: "```").first)
        try json.write(to: root.appendingPathComponent("workflow.json"), atomically: true, encoding: .utf8)
        let examples = try examples(in: document)
        struct Report: Decodable {
            let state: String
            let job_id: String
            let nodes: [Node]
            let outputs: [Output]
            struct Node: Decodable { let attempt: Int }
            struct Output: Decodable { let path: String }
        }
        for path in [["graph", "validate"], ["graph", "preflight"], ["graph", "run"]] {
            let example = try XCTUnwrap(examples.first { $0.arguments.starts(with: path) })
            let result = try runCLI(example.arguments, in: root)
            XCTAssertEqual(result.exit, 0, String(decoding: result.stderr, as: UTF8.self))
        }
        let inspect = try XCTUnwrap(examples.first { $0.arguments.starts(with: ["run", "inspect"]) })
        let inspection = try runCLI(inspect.arguments, in: root)
        XCTAssertEqual(inspection.exit, 0)
        let first = try JSONDecoder().decode(Report.self, from: inspection.stdout)
        XCTAssertEqual(first.state, "finished")
        let output = try XCTUnwrap(first.outputs.first)
        let artifact = root.appendingPathComponent("runs/message-check").appendingPathComponent(output.path)
        let value = try JSONDecoder().decode(String.self, from: Data(contentsOf: artifact))
        XCTAssertEqual(value, "Workflow execution verified.")

        let run = try XCTUnwrap(examples.first { $0.arguments.starts(with: ["graph", "run"]) })
        XCTAssertEqual(try runCLI(run.arguments + ["--resume"], in: root).exit, 0)
        let resumed = try JSONDecoder().decode(Report.self, from: runCLI(inspect.arguments, in: root).stdout)
        XCTAssertEqual(resumed.state, "finished")
        XCTAssertEqual(resumed.job_id, first.job_id)
        XCTAssertEqual(resumed.nodes.map(\.attempt), first.nodes.map(\.attempt), "Resume must reuse the finished node")
    }

    private func runCLI(_ arguments: [String], in directory: URL) throws -> (exit: Int32, stdout: Data, stderr: Data) {
        let binary = repositoryRoot.appendingPathComponent(".build/debug/mere.run")
        let outputURL = directory.appendingPathComponent("stdout-\(UUID().uuidString)")
        let errorURL = directory.appendingPathComponent("stderr-\(UUID().uuidString)")
        _ = FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        _ = FileManager.default.createFile(atPath: errorURL.path, contents: nil)
        let output = try FileHandle(forWritingTo: outputURL)
        let errors = try FileHandle(forWritingTo: errorURL)
        defer { try? output.close(); try? errors.close() }
        let process = Process()
        process.executableURL = binary
        process.currentDirectoryURL = directory
        process.arguments = ["--models-root", directory.appendingPathComponent("models").path] + arguments
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "MERERUN_API_KEY")
        environment["MERERUN_HUB_CACHE"] = directory.appendingPathComponent("hub").path
        process.environment = environment
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        let deadline = Date().addingTimeInterval(30)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.02) }
        guard !process.isRunning else {
            process.terminate()
            throw ValidationError("Skill command exceeded 30 seconds: \(arguments.joined(separator: " "))")
        }
        guard process.terminationReason == .exit else {
            throw ValidationError(
                "Skill command terminated by signal \(process.terminationStatus): \(arguments.joined(separator: " "))\n"
                    + String(decoding: try Data(contentsOf: errorURL), as: UTF8.self)
            )
        }
        return (process.terminationStatus, try Data(contentsOf: outputURL), try Data(contentsOf: errorURL))
    }

    private func matchingRequest(_ preflight: [String], execution: [String]) throws {
        let outputFlags: Set<String> = ["--preflight", "--json", "--receipt", "--progress-json", "--stream"]
        guard preflight.filter({ !outputFlags.contains($0) }) == execution.filter({ !outputFlags.contains($0) }) else {
            throw ValidationError("Execution changes the preflighted request: \(execution.joined(separator: " "))")
        }
    }

    private func checkDiscovery(_ command: ParsableCommand?, location: String) throws {
        if let catalog = command as? CatalogCommand, let identifier = catalog.id {
            XCTAssertNotNil(MereRunCapabilityCatalog.command(id: identifier), "\(location): unknown capability")
        }
        if let guide = command as? GuideCommand {
            if let model = guide.model {
                try checkModel(model, location: location)
            }
            if !guide.commandPath.isEmpty {
                let entry = try GuideCommand.resolveEntry(commandPath: guide.commandPath, model: guide.model)
                XCTAssertFalse(try GuideRegistry.content(for: entry, model: guide.model).isEmpty, location)
            } else if let model = guide.model {
                let handbook = try ModelGuideRegistry.guide(for: model)
                XCTAssertFalse(try GuideRegistry.content(for: handbook.entry).isEmpty, location)
            }
        }
    }

    private func checkModel(_ model: String, location: String) throws {
        _ = try XCTUnwrap(ManagedModelCatalog.allSpecs.first { $0.id == model }, "\(location): unknown model \(model)")
    }

    private func checkRecipe(_ command: ParsableCommand?, example: Example, pulledModels: Set<String>) throws {
        guard let model = try recipeModel(command, example: example) else { return }
        XCTAssertTrue(example.arguments.contains("--model"), "\(example.location): select the model explicitly")
        try checkModel(model, location: example.location)
        guard pulledModels.contains(model) else {
            throw ValidationError("\(example.location): recipe selects \(model) without a preceding pull for it")
        }
        if let server = command as? APIServe {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: model))
            XCTAssertEqual(spec.defaultRuntimeServingEngine, server.engine.runtimeServingEngine, example.location)
            XCTAssertEqual(server.defaultRuntimeModelID(modelPath: model), model, example.location)
        }
    }

    private func recipeModel(_ command: ParsableCommand?, example: Example) throws -> String? {
        switch command {
        case let image as ImageGenerate:
            guard let model = image.model else {
                throw ValidationError("\(example.location): image recipe must select its model explicitly")
            }
            return model
        case let chat as TextChat:
            return chat.model
        case let speech as SpeechSynthesize:
            return speech.model
        case let server as APIServe:
            return try XCTUnwrap(server.model, "\(example.location): API recipe must select its model explicitly")
        default:
            return nil
        }
    }

    private func parse(_ arguments: [String]) throws -> ParsableCommand? {
        var remaining = arguments[...]
        var commandType: ParsableCommand.Type = MereRunCLI.self
        while let name = remaining.first,
              let child = commandType.configuration.subcommands.first(where: { $0.configuration.commandName == name }) {
            commandType = child
            remaining = remaining.dropFirst()
        }
        // Only help/version examples may target the root, avoiding its bootstrap validation.
        if commandType == MereRunCLI.self {
            guard remaining == ["--help"] || remaining == ["--version"] else {
                throw ValidationError("Unknown root command in skill: \(arguments.joined(separator: " "))")
            }
            return nil
        }
        do {
            return try commandType.parseAsRoot(Array(remaining))
        } catch {
            if commandType.exitCode(for: error) == .success { return nil }
            throw error
        }
    }

    private func skillDocuments() throws -> [URL] {
        let root = repositoryRoot.appendingPathComponent("skills")
        return try FileManager.default.subpathsOfDirectory(atPath: root.path)
            .filter { $0.hasSuffix(".md") }.sorted().map { root.appendingPathComponent($0) }
    }

    private func examples(in document: URL) throws -> [Example] {
        let path = document.path
        let markdown = try String(contentsOf: document, encoding: .utf8)
        let lines = markdown.components(separatedBy: "\n")
        var inBash = false
        var block = 0
        var pending = ""
        var examples: [Example] = []
        for (offset, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                inBash = trimmed == "```bash"
                block += 1
                continue
            }
            guard inBash, !trimmed.isEmpty else { continue }
            if trimmed.hasSuffix("\\") {
                pending += trimmed.dropLast() + " "
                continue
            }
            let invocation = pending + trimmed
            pending = ""
            var words = try tokenize(invocation)
            if words.starts(with: ["swift", "run"]) { words.removeFirst(2) }
            guard words.first == "mere.run" else { continue }
            examples.append(Example(location: "\(path):\(offset + 1)", arguments: Array(words.dropFirst()), block: block))
        }
        XCTAssertTrue(pending.isEmpty, "\(path): unfinished command continuation")
        return examples
    }

    private func tokenize(_ command: String) throws -> [String] {
        // Skill examples use literal arguments and quoted prompts, with no shell evaluation.
        let pattern = #""[^"]*"|'[^']*'|[^\s"']+"#
        let expression = try NSRegularExpression(pattern: pattern)
        return try expression.matches(in: command, range: NSRange(command.startIndex..., in: command)).map { match in
            let range = try XCTUnwrap(Range(match.range, in: command))
            let word = String(command[range])
            if word.first == "\"" || word.first == "'" { return String(word.dropFirst().dropLast()) }
            return word
        }
    }
}
