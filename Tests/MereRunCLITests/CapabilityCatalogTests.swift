import ArgumentParser
import Foundation
import MereRunContract
import Testing

@testable import MereRunCLI

/// Decode the pinned ArgumentParser help format at the test boundary. Core and
/// the shared catalog remain independent of the parser and its metadata format.
private struct ParserHelp: Decodable {
    let serializationVersion: Int
    let command: Command

    struct Command: Decodable {
        let commandName: String
        let arguments: [Argument]?
        let subcommands: [Command]?
    }

    struct Argument: Decodable {
        enum Kind: String, Decodable { case positional, option, flag }
        let kind: Kind
        let shouldDisplay: Bool
        let isOptional: Bool
        let isRepeating: Bool
        let valueName: String?
        let names: [Name]?
        let defaultValue: String?
        let allValues: [String]?

        var flags: Set<String> {
            Set((names ?? []).filter { $0.kind == .long }.map { "--" + $0.name })
        }
    }

    struct Name: Decodable {
        enum Kind: String, Decodable { case long, short, longWithSingleDash }
        let kind: Kind
        let name: String
    }
}

private func parserCommands() throws -> [String: ParserHelp.Command] {
    let help = try JSONDecoder().decode(ParserHelp.self, from: Data(MereRunCLI._dumpHelp().utf8))
    #expect(help.serializationVersion == 0, "Review the parser adapter before adopting a new help format.")
    var commands: [String: ParserHelp.Command] = [:]
    func walk(_ command: ParserHelp.Command, path: [String]) {
        for child in command.subcommands ?? [] {
            let next = path + [child.commandName]
            if child.subcommands?.isEmpty ?? true {
                commands[next.joined(separator: ".")] = child
            } else {
                walk(child, path: next)
            }
        }
    }
    walk(help.command, path: [])
    return commands
}

@Test func capabilityOptionsMatchArgumentParser() throws {
    let commands = try parserCommands()
    for capability in MereRunCapabilityCatalog.document.commands {
        let command = try #require(commands[capability.id], "Unrecognized capability: \(capability.id)")
        let arguments = (command.arguments ?? []).filter { $0.shouldDisplay }
        // ArgumentParser adds these transport-wide flags to every leaf.
        let builtins: Set<String> = ["--help", "--version"]
        let options = arguments.filter { $0.kind != .positional && $0.flags.isDisjoint(with: builtins) }
        let declared = Set(capability.options.map(\.flag))
        let parsed = options.reduce(into: Set<String>()) { $0.formUnion($1.flags) }
        #expect(declared.isSubset(of: parsed), "\(capability.id) declares unknown flags: \(declared.subtracting(parsed))")
        #expect(capability.command.joined(separator: ".") == capability.id)
        for option in options {
            // Long aliases share a value. An inverted Boolean has two values,
            // so require both polarities (with any equivalent spelling).
            let negative: Set<String> = option.kind == .flag ? option.flags.filter { $0.hasPrefix("--no-") } : []
            let positive = option.flags.subtracting(negative)
            for aliases in [positive, negative] where !aliases.isEmpty {
                #expect(!declared.isDisjoint(with: aliases), "\(capability.id) omits \(aliases.sorted())")
            }
        }
        for option in capability.options {
            let parsed = try #require(options.first { $0.flags.contains(option.flag) })
            let context = "\(capability.id) \(option.flag)"
            #expect(option.required == !parsed.isOptional, "\(context): required")
            #expect(option.repeatable == parsed.isRepeating, "\(context): repeatable")
            #expect((option.kind == .boolean) == (parsed.kind == .flag), "\(context): flag or value")
            if let value = option.defaultValue {
                #expect(value == parsed.defaultValue, "\(context): default")
            }
            if option.kind == .choice, let values = parsed.allValues {
                #expect(Set(option.choices) == Set(values), "\(context): choices")
            }
        }
        if let flag = capability.output.flag {
            #expect(parsed.contains(flag), "\(capability.id) output flag is not parsed: \(flag)")
        }
    }
}

/// Array positionals can be empty at the parser boundary yet be required by the
/// command's validation. Keep these semantic requirements explicit and narrow.
private let requiredPositionalOverrides: [String: String] = [
    "text.embed.texts": "Requires text unless --list-models is set.",
    "vision.caption.images": "Requires at least one image.",
    "vision.ocr.images": "Requires at least one image.",
    "vision.geometry-multiview.images": "Requires multiple images for geometry reconstruction."
]

/// Preserve existing catalog keys used by stored shell drafts.
private let positionalNameAliases: [String: String] = [
    "model.location.bind.modelID": "model-id",
    "model.location.unbind.modelID": "model-id"
]

@Test func capabilityPositionalsMatchArgumentParser() throws {
    let commands = try parserCommands()
    var usedOverrides: Set<String> = []
    var usedAliases: Set<String> = []
    for capability in MereRunCapabilityCatalog.document.commands {
        let command = try #require(commands[capability.id])
        let parsed = (command.arguments ?? []).filter { $0.shouldDisplay && $0.kind == .positional }
        #expect(capability.arguments.count == parsed.count, "\(capability.id): positional count")
        for (argument, parser) in zip(capability.arguments, parsed) {
            let key = "\(capability.id).\(argument.name)"
            let name = positionalNameAliases[key] ?? argument.name
            if positionalNameAliases[key] != nil { usedAliases.insert(key) }
            let required = requiredPositionalOverrides[key] != nil || !parser.isOptional
            if requiredPositionalOverrides[key] != nil { usedOverrides.insert(key) }
            #expect(name == parser.valueName, "\(key): positional name or order")
            #expect(argument.required == required, "\(key): required")
            #expect(argument.repeatable == parser.isRepeating, "\(key): repeatable")
        }
    }
    #expect(usedOverrides == Set(requiredPositionalOverrides.keys), "Remove stale positional requirements.")
    #expect(usedAliases == Set(positionalNameAliases.keys), "Remove stale positional aliases.")
}

@Test func catalogCommandParsesASelectedCapability() throws {
    let command = try CatalogCommand.parse(["video.generate", "--json"])
    #expect(command.id == "video.generate")
    #expect(command.json)
    #expect(MereRunCapabilityCatalog.command(id: command.id ?? "")?.id == "video.generate")
}

/// Every public CLI leaf command must either be described by the shared
/// capability contract or appear in `contractExemptCommandIDs` with the reason
/// it is deliberately absent. Without this the CLI can grow a command that no
/// shell ever surfaces: `capabilityOptionsMatchArgumentParser` only walks the
/// contract, and the app's inverse coverage test is keyed to the contract too,
/// so an uncataloged command is invisible to both.
let contractExemptCommandIDs: [String: String] = [
    "catalog": "Emits the contract itself; shells compile against MereRunContract instead.",
    "relay.serve": "Relay console owns the control plane. See apps/macos/README.md.",
    "executor.add.ssh": "Relay console owns executor profiles.",
    "executor.add.relay": "Relay console owns executor profiles.",
    "executor.list": "Relay console owns executor profiles.",
    "executor.inspect": "Relay console owns executor profiles.",
    "executor.probe": "Relay console owns executor profiles.",
    "executor.login": "Relay console owns device sign-in.",
    "executor.auth-status": "Relay console owns device sign-in.",
    "executor.logout": "Relay console owns device sign-in.",
    "executor.fleet": "Relay console owns fleet telemetry.",
    "executor.node-refresh": "Relay console owns node lifecycle.",
    "executor.node-configure": "Relay console owns scheduling policy.",
    "executor.remove": "Relay console owns executor profiles.",
    "graph.catalog": "Graph Studio owns workflow authoring.",
    "graph.dataset.discover": "Graph Studio owns workflow authoring.",
    "graph.validate": "Graph Studio owns workflow authoring.",
    "graph.preflight": "Graph Studio owns workflow authoring.",
    "graph.materialize": "Graph Studio owns workflow authoring.",
    "graph.export-job": "Graph Studio owns workflow authoring.",
    "graph.run": "Graph Studio owns workflow execution.",
    "graph.run-job": "Graph Studio owns workflow execution.",
    "graph.submit": "Graph Studio owns workflow execution.",
    "graph.submit-job": "Graph Studio owns workflow execution.",
    "graph.worker.probe": "Machine-to-machine worker protocol, not a shell surface.",
    "graph.worker.execute": "Machine-to-machine worker protocol, not a shell surface.",
    "graph.worker.inspect": "Machine-to-machine worker protocol, not a shell surface.",
    "graph.worker.cancel": "Machine-to-machine worker protocol, not a shell surface.",
    "model.benchmark.q38-verification": "Research-only target-verification microbenchmark, not a product workflow.",
    "vision.image-to-3d": "VFX alias of image.reconstruct-3d, surfaced through the Image workspace.",
    "vision.image-to-3d-trellis2": "VFX alias of image.reconstruct-3d-trellis2.",
    "vision.image-to-3d-multiview": "VFX alias of image.reconstruct-3d-multiview."
]

@Test func everyPublicCLICommandIsCatalogedOrExplicitlyExempt() throws {
    let cataloged = Set(MereRunCapabilityCatalog.document.commands.map(\.id))
    let leaves = try parserCommands().keys

    for id in leaves {
        #expect(
            cataloged.contains(id) || contractExemptCommandIDs[id] != nil,
            """
            `mere.run \(id.replacingOccurrences(of: ".", with: " "))` is not in the shared \
            capability contract. Add a MereRunCommandCapability for it so shells can surface \
            it, or add it to contractExemptCommandIDs with the reason it stays CLI-only.
            """
        )
    }

    let leafIDs = Set(leaves)
    for (id, reason) in contractExemptCommandIDs {
        #expect(leafIDs.contains(id), "Exemption \(id) no longer matches a CLI command: \(reason)")
        #expect(!cataloged.contains(id), "\(id) is cataloged now; remove its exemption.")
    }
}
