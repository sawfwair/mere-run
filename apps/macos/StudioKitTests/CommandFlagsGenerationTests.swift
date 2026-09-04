@testable import StudioKit
import MereRunContract
import XCTest

/// Keeps the generated flag constants in step with the shared contract.
///
/// `CommandFlags.swift` is written from `MereRunCapabilityCatalog`, so every flag the app
/// emits is a symbol the contract declares. This test regenerates the file and compares it
/// with what is committed; run `./scripts/update-studio-command-flags.sh` to re-record after
/// changing a capability's options.
final class CommandFlagsGenerationTests: XCTestCase {
    func testGeneratedFlagConstantsMatchTheContract() throws {
        let rendered = CommandFlagsGenerator.render(MereRunCapabilityCatalog.document)
        if ProcessInfo.processInfo.environment["MERERUN_UPDATE_STUDIO_FLAGS"] == "1" {
            try rendered.write(to: CommandFlagsGenerator.url, atomically: true, encoding: .utf8)
            return
        }

        let committed = try String(contentsOf: CommandFlagsGenerator.url, encoding: .utf8)
        guard rendered != committed else { return }

        let renderedLines = rendered.split(separator: "\n", omittingEmptySubsequences: false)
        let committedLines = committed.split(separator: "\n", omittingEmptySubsequences: false)
        for (index, committedLine) in committedLines.enumerated() where index < renderedLines.count {
            guard renderedLines[index] != committedLine else { continue }
            XCTFail(
                """
                CommandFlags.swift is out of date at line \(index + 1): the contract renders \
                "\(renderedLines[index])" where the committed file has "\(committedLine)". \
                Run ./scripts/update-studio-command-flags.sh.
                """
            )
            return
        }
        XCTFail("CommandFlags.swift is out of date; run ./scripts/update-studio-command-flags.sh")
    }

    /// Every capability must render a namespace whose members are unique and spellable, so a
    /// new contract option cannot silently collide with an existing constant.
    func testEveryCapabilityRendersDistinctConstantNames() throws {
        for capability in MereRunCapabilityCatalog.document.commands {
            var seen: Set<String> = ["command", "defaultValues"]
            for option in capability.options {
                let name = CommandFlagsGenerator.constantName(for: option.flag)
                XCTAssertFalse(
                    seen.contains(name),
                    "\(capability.id) renders \(option.flag) as \(name), which is already taken"
                )
                seen.insert(name)
            }
        }
    }
}

/// Writes `CommandFlags.swift` from the shared capability contract.
enum CommandFlagsGenerator {
    static let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("StudioKit/Catalog/CommandFlags.swift")

    /// Segments that read as acronyms, so `--progress-json` becomes `progressJSON` rather
    /// than `progressJson`.
    private static let acronyms = [
        "3d": "3D", "ae": "AE", "api": "API", "dflash": "DFlash", "id": "ID", "json": "JSON",
        "jsonl": "JSONL", "kv": "KV", "lm": "LM", "lora": "LoRA", "mtp": "MTP", "ocr": "OCR",
        "olmoearth": "OlmoEarth", "sfx": "SFX", "ttl": "TTL", "url": "URL", "vae": "VAE",
        "vlm": "VLM", "webui": "WebUI"
    ]

    /// Swift keywords a flag could land on; a constant named after one is back-ticked.
    private static let keywords: Set<String> = [
        "as", "associatedtype", "break", "case", "catch", "class", "continue", "default",
        "defer", "deinit", "do", "else", "enum", "extension", "fallthrough", "false", "for",
        "func", "guard", "if", "import", "in", "init", "inout", "internal", "is", "let", "nil",
        "operator", "private", "protocol", "public", "repeat", "rethrows", "return", "self",
        "static", "struct", "subscript", "super", "switch", "throw", "throws", "true", "try",
        "typealias", "var", "where", "while"
    ]

    static func constantName(for flag: String) -> String {
        let segments = flag.drop(while: { $0 == "-" }).split(separator: "-").map(String.init)
        guard let head = segments.first else { return flag }
        let name = head + segments.dropFirst().map { acronyms[$0] ?? $0.capitalizedFirst }.joined()
        return keywords.contains(name) ? "`\(name)`" : name
    }

    static func namespaceName(for capabilityID: String) -> String {
        capabilityID
            .split(whereSeparator: { $0 == "." || $0 == "-" })
            .map { acronyms[String($0)] ?? String($0).capitalizedFirst }
            .joined()
    }

    static func render(_ document: MereRunCapabilityDocument) -> String {
        var lines = [
            "// Generated from MereRunCapabilityCatalog by",
            "// StudioKitTests/CommandFlagsGenerationTests.swift. Do not edit by hand:",
            "// run ./scripts/update-studio-command-flags.sh after changing the shared contract.",
            "",
            "/// Every flag the shared contract declares, as a constant per capability.",
            "///",
            "/// `CommandArguments` builds argv from these instead of string literals, so a flag the",
            "/// CLI renames or drops is a compile error in the app rather than a command line that",
            "/// only fails when it runs.",
            "package enum CommandFlags {}",
            ""
        ]

        for capability in document.commands {
            let path = capability.command.joined(separator: " ")
            lines.append("// MARK: - \(path)")
            lines.append("")
            lines.append("extension CommandFlags {")
            lines.append("    /// `mere.run \(path)` — \(capability.title)")
            lines.append("    package enum \(namespaceName(for: capability.id)): CommandFlagNamespace {")
            let command = capability.command.map { "\"\($0)\"" }.joined(separator: ", ")
            lines.append("        package static let command = [\(command)]")

            let defaults = capability.options.compactMap { option in
                option.defaultValue.map { (option.flag, $0) }
            }
            if !defaults.isEmpty {
                lines.append("        package static let defaultValues = [")
                for (index, entry) in defaults.enumerated() {
                    let comma = index == defaults.count - 1 ? "" : ","
                    lines.append("            \"\(entry.0)\": \"\(entry.1)\"\(comma)")
                }
                lines.append("        ]")
            }
            if !capability.options.isEmpty {
                lines.append("")
            }
            for option in capability.options {
                lines.append("        package static let \(constantName(for: option.flag)) = \"\(option.flag)\"")
            }
            lines.append("    }")
            lines.append("}")
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }
}

private extension String {
    var capitalizedFirst: String {
        guard let first else { return self }
        return first.uppercased() + dropFirst()
    }
}
