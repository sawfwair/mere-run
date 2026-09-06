import ArgumentParser
import Foundation
import MereRunContract
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
        for skill in ["mere-run", "use-mere-run"] {
            let examples = try examples(in: skill)
            XCTAssertFalse(examples.isEmpty, "No CLI examples found in \(skill)")
            var pulledModels: [Int: String] = [:]
            for example in examples {
                do {
                    // Parse the leaf directly: root validation bootstraps machine-local state.
                    // Never invoke run(), model resolution, or a shell for documentation checks.
                    let command = try parse(example.arguments)
                    if let pull = command as? ModelPull {
                        let model = try XCTUnwrap(pull.target, example.location)
                        try checkModel(model, location: example.location)
                        pulledModels[example.block] = model
                    }
                    try checkDiscovery(command, location: example.location)
                    try checkRecipe(command, example: example, pulledModel: pulledModels[example.block])
                } catch {
                    XCTFail("\(example.location): \(error)")
                }
            }
        }
    }

    func testRecipeCheckRejectsImplicitOrMismatchedImageModels() throws {
        let example = Example(location: "regression fixture", arguments: [], block: 0)
        let implicit = try ImageGenerate.parse(["--prompt", "a mug"])
        let different = try ImageGenerate.parse(["--prompt", "a mug", "--model", "image-zimage-nano"])
        XCTAssertThrowsError(try recipeModel(implicit, example: example))
        XCTAssertThrowsError(try matchingModel("image-zimage-nano", pulledModel: "image-klein-max"))
        XCTAssertEqual(try recipeModel(different, example: example), "image-zimage-nano")
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

    private func checkRecipe(_ command: ParsableCommand?, example: Example, pulledModel: String?) throws {
        guard let model = try recipeModel(command, example: example) else { return }
        XCTAssertTrue(example.arguments.contains("--model"), "\(example.location): select the model explicitly")
        try checkModel(model, location: example.location)
        let pulled = try XCTUnwrap(pulledModel, "\(example.location): recipe needs its model pull in the same block")
        try matchingModel(model, pulledModel: pulled)
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

    private func matchingModel(_ selected: String, pulledModel: String) throws {
        guard selected == pulledModel else {
            throw ValidationError("Recipe pulls \(pulledModel) but selects \(selected)")
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

    private func examples(in skill: String) throws -> [Example] {
        let path = "skills/\(skill)/SKILL.md"
        let markdown = try String(contentsOf: repositoryRoot.appendingPathComponent(path), encoding: .utf8)
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
