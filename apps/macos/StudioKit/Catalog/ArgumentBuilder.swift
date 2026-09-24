import Foundation

/// One capability's contract-declared flags, generated into `CommandFlags` from
/// `MereRunCapabilityCatalog`.
///
/// Every flag the app emits is a constant on one of these namespaces, so a flag the CLI
/// renames or drops stops compiling here instead of failing a test at runtime.
package protocol CommandFlagNamespace {
    /// The subcommand path the capability is invoked with, for example `["image", "generate"]`.
    static var command: [String] { get }
    /// The CLI's own default for the options that declare one, keyed by flag.
    static var defaultValues: [String: String] { get }
}

extension CommandFlagNamespace {
    package static var defaultValues: [String: String] { [:] }
}

/// Builds one CLI command line out of a capability's flags.
///
/// The builder starts from the capability's subcommand path and appends positional
/// arguments, switches, and `--flag value` pairs in call order, so the argv it returns reads
/// the way a person would type it.
package struct ArgumentBuilder {
    package private(set) var arguments: [String]
    private let defaultValues: [String: String]

    /// Starts a command line for a generated flag namespace.
    package init<Flags: CommandFlagNamespace>(_ flags: Flags.Type) {
        arguments = Flags.command
        defaultValues = Flags.defaultValues
    }

    /// Starts a command line the shared contract does not describe.
    package init(_ command: [String]) {
        arguments = command
        defaultValues = [:]
    }

    /// Appends a positional argument.
    package mutating func value(_ value: String) {
        arguments.append(value)
    }

    /// Appends several positional arguments.
    package mutating func values(_ values: [String]) {
        arguments.append(contentsOf: values)
    }

    /// Appends a switch that carries no value.
    package mutating func flag(_ flag: String) {
        arguments.append(flag)
    }

    /// Appends one half of a `--x` / `--no-x` pair, or neither when the control leaves the
    /// choice to the CLI.
    package mutating func pair(_ enabled: String, _ disabled: String, _ isEnabled: Bool?) {
        guard let isEnabled else { return }
        arguments.append(isEnabled ? enabled : disabled)
    }

    /// Appends `--flag value`, or `--flag=value` when the value starts with a dash.
    package mutating func option(_ flag: String, _ value: String) {
        arguments += Self.optionArguments(flag, value)
    }

    /// `[flag, value]`, or the one joined `flag=value` argument when the value starts with a
    /// dash. ArgumentParser reads a separate `-1` after an option as the next option and fails
    /// with "Missing value", so a negative number (a peak of -1 dBFS, a MIDI transposition of
    /// -12) only reaches the CLI in the joined form. Every place that writes a flag and its value
    /// into argv goes through here; every reader splits a token on its first `=`.
    package static func optionArguments(_ flag: String, _ value: String) -> [String] {
        value.hasPrefix("-") ? ["\(flag)=\(value)"] : [flag, value]
    }

    /// The flag a token spells and the value joined to it: `--flag=value` reads as
    /// `("--flag", "value")`, `--flag` as `("--flag", nil)`, and a value or a subcommand as nil.
    package static func splitOption(_ token: String) -> (flag: String, value: String?)? {
        guard token.hasPrefix("--") else { return nil }
        let parts = token.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
        return (String(parts[0]), parts.count == 2 ? String(parts[1]) : nil)
    }

    /// Appends `--flag value` once per value, for the options the contract marks repeatable.
    package mutating func repeated(_ flag: String, _ values: [String]) {
        for value in values {
            option(flag, value)
        }
    }

    /// Appends `--flag value` unless the value is what the CLI would use anyway.
    ///
    /// The comparison is against the contract's `default_value`, so the app never has to
    /// restate a CLI default as a literal in order to leave it off the command line.
    package mutating func optionUnlessDefault(_ flag: String, _ value: String) {
        guard defaultValues[flag] != value else { return }
        option(flag, value)
    }
}
