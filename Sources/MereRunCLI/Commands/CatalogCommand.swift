import ArgumentParser
import Foundation
import MereRunContract

struct CatalogCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "catalog",
        abstract: "Inspect the machine-readable command capability contract."
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
                print(try encode(capability))
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
            print(try encode(MereRunCapabilityCatalog.document))
        } else {
            for capability in MereRunCapabilityCatalog.document.commands {
                print("\(capability.id)\t\(capability.title)")
            }
        }
    }

    private func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        guard let result = String(data: data, encoding: .utf8) else {
            throw ValidationError("Could not encode the capability catalog as UTF-8.")
        }
        return result
    }
}
