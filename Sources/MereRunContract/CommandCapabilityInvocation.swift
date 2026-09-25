import Foundation

/// The arguments after a command path, read the way ArgumentParser reads them for one
/// capability: each declared option's values in order under its canonical flag (a Boolean maps
/// to `[]`), the positional values, and the option tokens the contract does not declare.
/// Aliases and `--flag=value` fold into the canonical flag; a value-taking option consumes the
/// next token whatever it looks like, so `--target-peak-db -1` reads as one value.
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
        while index < arguments.endIndex {
            let token = arguments[index]
            index += 1
            if token == "--" {
                positionals.append(contentsOf: arguments[index...])
                break
            }
            guard token.hasPrefix("-"), token.count > 1 else {
                positionals.append(token)
                continue
            }
            let (name, inlineValue) = Self.split(token)
            guard let option = spellings[name] else {
                undeclared.append(token)
                continue
            }
            var occurrences = values[option.flag, default: []]
            if option.kind != .boolean {
                if let inlineValue {
                    occurrences.append(inlineValue)
                } else if index < arguments.endIndex {
                    occurrences.append(arguments[index])
                    index += 1
                }
            }
            values[option.flag] = occurrences
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

    private static func split(_ token: String) -> (name: String, value: String?) {
        guard let equals = token.firstIndex(of: "=") else { return (token, nil) }
        return (String(token[..<equals]), String(token[token.index(after: equals)...]))
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
