import Foundation

/// The arguments after a command path, read the way ArgumentParser reads them for one
/// capability: each declared option's values under its canonical flag (a Boolean maps to `[]`),
/// the positional values, and the option tokens the contract does not declare.
///
/// - Aliases, `--flag=value`, and `-f=value` fold into the canonical flag.
/// - A value-taking option takes the next token only when that token is a value: anything but
///   an option-shaped token (`-x`, `--x`). ArgumentParser refuses `--target-peak-db -1` as a
///   missing value, so the reader records no value there and reads `-1` as an undeclared option;
///   `--target-peak-db=-1` is the spelling that passes a negative number.
/// - A single-dash group expands into its short options (`-qm song` is `-q -m song`), and each
///   value-taking option in the group takes the next value in order.
/// - A single-value option keeps only its last occurrence, as ArgumentParser does; a repeatable
///   option keeps every one in order.
public struct MereRunCommandInvocation: Equatable, Sendable {
    public let values: [String: [String]]
    public let positionals: [String]
    public let undeclared: [String]

    public init(capability: MereRunCommandCapability, arguments: [String]) {
        var spellings: [String: MereRunCapabilityOption] = [:]
        for option in capability.options {
            for spelling in option.spellings {
                spellings[spelling] = option
            }
        }
        var values: [String: [String]] = [:]
        var positionals: [String] = []
        var undeclared: [String] = []
        var index = arguments.startIndex
        // Takes the next token as a value when ArgumentParser would.
        func nextValue() -> String? {
            guard index < arguments.endIndex, Self.isValue(arguments[index]) else { return nil }
            defer { index += 1 }
            return arguments[index]
        }
        func record(_ option: MereRunCapabilityOption, value: String?) {
            guard option.kind != .boolean else {
                values[option.flag] = values[option.flag] ?? []
                return
            }
            // A value-taking option without a value is ArgumentParser's parse error; the command
            // never runs, so nothing is recorded for it.
            guard let value else { return }
            values[option.flag] = option.repeatable ? values[option.flag, default: []] + [value] : [value]
        }
        while index < arguments.endIndex {
            let token = arguments[index]
            index += 1
            if token == "--" {
                positionals.append(contentsOf: arguments[index...])
                break
            }
            guard !Self.isValue(token) else {
                positionals.append(token)
                continue
            }
            let (name, inlineValue) = Self.split(token)
            if let option = spellings[name] {
                record(option, value: option.kind == .boolean ? nil : inlineValue ?? nextValue())
                continue
            }
            guard let group = Self.shortGroup(token) else {
                undeclared.append(token)
                continue
            }
            for short in group {
                guard let option = spellings[short] else {
                    undeclared.append(short)
                    continue
                }
                record(option, value: option.kind == .boolean ? nil : nextValue())
            }
        }
        self.values = values
        self.positionals = positionals
        self.undeclared = undeclared
    }

    /// True when the invocation passes `flag` (a Boolean is on).
    public func contains(_ flag: String) -> Bool {
        values[flag] != nil
    }

    /// The value ArgumentParser keeps for a single-value option: the last one passed.
    public func value(_ flag: String) -> String? {
        values[flag]?.last
    }

    /// ArgumentParser reads a token as a value unless it starts with a dash; a lone `-` (stdin)
    /// and an empty string are values.
    static func isValue(_ token: String) -> Bool {
        !token.hasPrefix("-") || token == "-"
    }

    private static func split(_ token: String) -> (name: String, value: String?) {
        guard let equals = token.firstIndex(of: "=") else { return (token, nil) }
        return (String(token[..<equals]), String(token[token.index(after: equals)...]))
    }

    /// `-qm` as ArgumentParser splits it: one short option per letter or digit. `nil` for a long
    /// option, a single short option, or a `-name=value` token.
    private static func shortGroup(_ token: String) -> [String]? {
        guard !token.hasPrefix("--"), !token.contains("="), token.count > 2 else { return nil }
        let letters = token.dropFirst()
        guard letters.allSatisfy({ $0.isLetter || $0.isNumber }) else { return nil }
        return letters.map { "-\($0)" }
    }
}

extension MereRunCapabilityCatalog {
    /// The capability a command line runs and the arguments after its command path, matched on
    /// the longest `command` prefix of the leading non-option tokens. `commandLine` excludes the
    /// executable. `nil` for commands the contract does not describe.
    public static func capability(
        forCommandLine commandLine: [String]
    ) -> (capability: MereRunCommandCapability, arguments: [String])? {
        let path = commandLine.prefix { !$0.hasPrefix("-") }
        let match = document.commands
            .filter { path.starts(with: $0.command) }
            .max { $0.command.count < $1.command.count }
        return match.map { ($0, Array(commandLine.dropFirst($0.command.count))) }
    }
}
