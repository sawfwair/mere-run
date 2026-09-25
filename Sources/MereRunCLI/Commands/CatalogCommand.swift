import ArgumentParser
import Foundation
import MereRunContract

struct CatalogCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "catalog",
        abstract: "Inspect the machine-readable command capability contract.",
        subcommands: [CatalogShowCommand.self, CatalogResolveCommand.self],
        defaultSubcommand: CatalogShowCommand.self
    )

    static func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        guard let result = String(data: data, encoding: .utf8) else {
            throw ValidationError("Could not encode the capability catalog as UTF-8.")
        }
        return result
    }
}

struct CatalogShowCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "show",
        abstract: "Print the capability contract, or one capability by id (the default)."
    )

    @Argument(help: "Optional capability id, for example video.generate.")
    var id: String?

    @Flag(name: [.long], help: "Emit stable JSON.")
    var json = false

    func run() throws {
        if let id {
            guard let capability = MereRunCapabilityCatalog.command(id: id) else {
                let known = MereRunCapabilityCatalog.document.commands.map(\.id).joined(separator: ", ")
                throw ValidationError("Unknown capability '\(id)'. Known ids: \(known)")
            }
            if json {
                print(try CatalogCommand.encode(capability))
            } else {
                print("\(capability.id): \(capability.title)")
                print(capability.summary)
                print("command: \(capability.command.joined(separator: " "))")
                for option in capability.options {
                    let choices = option.choices.isEmpty
                        ? ""
                        : " [\(option.choices.joined(separator: "|"))]"
                    print("  \(option.flag)\(choices) — \(option.label)")
                }
            }
            return
        }

        if json {
            print(try CatalogCommand.encode(MereRunCapabilityCatalog.document))
        } else {
            for capability in MereRunCapabilityCatalog.document.commands {
                print("\(capability.id)\t\(capability.title)")
            }
        }
    }
}

struct CatalogResolveCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "resolve",
        abstract: "Show which runtime family a command line would run, and the options it rejects or ignores.",
        discussion: """
        Pass the command line after `--`, without `mere.run`:

          mere.run catalog resolve --json -- music analyze song.wav --model music-acestep

        Nothing is loaded or downloaded. `violations` lists why the command would refuse to run \
        and `warnings` lists options it would run without; both empty means it runs as given.
        """
    )

    @Flag(name: [.long], help: "Emit stable JSON.")
    var json = false

    @Argument(parsing: .postTerminator, help: "The command line to resolve, after `--`.")
    var commandLine: [String]

    func validate() throws {
        guard !commandLine.isEmpty else {
            throw ValidationError("Pass the command line after `--`, for example: catalog resolve -- music analyze song.wav")
        }
    }

    func run() throws {
        guard let (_, report) = CLICapabilityGate.evaluate(commandLine: commandLine) else {
            throw ValidationError("`\(commandLine.joined(separator: " "))` is not a command in the capability catalog.")
        }
        if json {
            print(try CatalogCommand.encode(report))
            return
        }
        print("capability: \(report.capability)")
        if let family = report.family {
            print("family: \(family)\(report.familyTitle.map { " (\($0))" } ?? "")")
        }
        if let model = report.model {
            print("model: \(model)")
        }
        print("source: \(report.source.rawValue)")
        for violation in report.violations {
            print("error: \(violation)")
        }
        for warning in report.warnings {
            print("warning: \(warning)")
        }
    }
}
