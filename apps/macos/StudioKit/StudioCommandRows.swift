import Foundation
import MereRunContract

// The Command view's rows come from the exact argv the task's template emits for the current
// draft (`CommandTemplate.arguments(from:)`), annotated with the contract's option labels and
// kinds. Nothing here builds a command of its own, so the column can never disagree with what
// Run launches.

/// The eyebrow a row files under: the contract's own `group` for every option a capability
/// declares, `Arguments` for positionals, and `Options` for a flag the contract has yet to
/// describe. The app keeps no grouping table of its own — the inspector's sections and this
/// column's eyebrows are the same list, read from the same place.
package enum StudioCommandRowGroup: Hashable, Identifiable {
    case arguments
    case contract(StudioContractGroup)

    package var id: String { title }

    package var title: String {
        switch self {
        case .arguments: return "Arguments"
        case .contract(let group): return group.title
        }
    }

    /// Positionals first, then the contract's groups in the order both columns show them.
    package static let allCases: [StudioCommandRowGroup] = [.arguments]
        + StudioContractGroup.allCases.map { .contract($0) }

    package static func group(forFlag flag: String, in capability: MereRunCommandCapability?) -> StudioCommandRowGroup {
        .contract(StudioContractGroup.group(forFlag: flag, in: capability))
    }
}

/// One line of the raw form: a flag and what the draft gives it.
package struct StudioCommandRow: Identifiable, Equatable {
    package enum Value: Equatable {
        /// A flag with a value; empty when the draft leaves the option unset.
        case text(String)
        /// A boolean flag: on when the argv carries it.
        case toggle(Bool)
        /// A positional argument.
        case positional(String)
    }

    package let flag: String
    /// The contract's label for the option; the flag itself when the contract has none.
    package let label: String
    package let value: Value

    package var id: String { flag }

    package var isSet: Bool {
        switch value {
        case .text(let text): return !text.isEmpty
        case .toggle(let on): return on
        case .positional: return true
        }
    }
}

package struct StudioCommandRowGroupRows: Identifiable, Equatable {
    package let group: StudioCommandRowGroup
    package let rows: [StudioCommandRow]

    package var id: StudioCommandRowGroup { group }
}

package enum StudioCommandRows {
    /// The argv split into positional arguments and `(flag, value?)` pairs, after the command
    /// path. A flag followed by another flag (or by nothing) carries no value.
    package static func parse(arguments: [String], commandPathCount: Int) -> (positional: [String], flags: [(String, String?)]) {
        var positional: [String] = []
        var flags: [(String, String?)] = []
        var index = min(commandPathCount, arguments.count)
        while index < arguments.count {
            let token = arguments[index]
            if token.hasPrefix("--") {
                let next = index + 1 < arguments.count ? arguments[index + 1] : nil
                if let next, !next.hasPrefix("--") {
                    flags.append((token, next))
                    index += 2
                } else {
                    flags.append((token, nil))
                    index += 1
                }
            } else {
                positional.append(token)
                index += 1
            }
        }
        return (positional, flags)
    }

    /// The grouped rows for `template` and `draft`: every option the contract declares (set
    /// ones first within a group, then unset in declaration order), plus any emitted flag the
    /// contract does not list, plus positional arguments. Repeated flags join with newlines.
    package static func groups(template: CommandTemplate, draft: CommandDraft) -> [StudioCommandRowGroupRows] {
        let arguments = template.arguments(from: draft)
        let capability = template.id.capabilityID.flatMap { MereRunCapabilityCatalog.command(id: $0) }
        let commandPathCount = capability?.command.count ?? Self.commandPathCount(of: arguments)
        let parsed = parse(arguments: arguments, commandPathCount: commandPathCount)
        let declared = capability?.options ?? []
        let declaredByFlag = Dictionary(declared.map { ($0.flag, $0) }, uniquingKeysWith: { first, _ in first })

        var emitted: [String: [String?]] = [:]
        var emittedOrder: [String] = []
        for (flag, value) in parsed.flags {
            if emitted[flag] == nil { emittedOrder.append(flag) }
            emitted[flag, default: []].append(value)
        }

        var rowsByGroup: [StudioCommandRowGroup: [StudioCommandRow]] = [:]
        func append(_ row: StudioCommandRow) {
            rowsByGroup[StudioCommandRowGroup.group(forFlag: row.flag, in: capability), default: []].append(row)
        }

        for option in declared {
            let values = emitted[option.flag]
            let value: StudioCommandRow.Value
            if option.kind == .boolean {
                value = .toggle(values != nil)
            } else {
                value = .text((values ?? []).compactMap { $0 }.joined(separator: "\n"))
            }
            append(StudioCommandRow(flag: option.flag, label: option.label, value: value))
        }
        for flag in emittedOrder where declaredByFlag[flag] == nil {
            let values = (emitted[flag] ?? []).compactMap { $0 }
            let value: StudioCommandRow.Value = values.isEmpty ? .toggle(true) : .text(values.joined(separator: "\n"))
            append(StudioCommandRow(flag: flag, label: flag, value: value))
        }

        var groups: [StudioCommandRowGroupRows] = []
        if !parsed.positional.isEmpty {
            let labels = capability?.arguments.map(\.label) ?? []
            let rows = parsed.positional.enumerated().map { index, value in
                let label = index < labels.count ? labels[index] : "argument \(index + 1)"
                return StudioCommandRow(flag: label, label: label, value: .positional(value))
            }
            groups.append(StudioCommandRowGroupRows(group: .arguments, rows: rows))
        }
        for group in StudioCommandRowGroup.allCases where group != .arguments {
            guard let rows = rowsByGroup[group], !rows.isEmpty else { continue }
            let ordered = rows.filter(\.isSet) + rows.filter { !$0.isSet }
            groups.append(StudioCommandRowGroupRows(group: group, rows: ordered))
        }
        return groups
    }

    /// Leading non-flag tokens: the subcommand path of a template without a contract entry.
    private static func commandPathCount(of arguments: [String]) -> Int {
        arguments.prefix { !$0.hasPrefix("--") }.count
    }
}

/// Lays the shell-quoted display command out the way the board shows it: the executable and
/// command path on the first line, then flag/value pairs packed up to `width` columns, each
/// continued line ending in a backslash.
package enum StudioCommandPreviewFormatter {
    package static func wrapped(_ displayCommand: String, width: Int = 72) -> String {
        let words = ShellWords.split(displayCommand)
        guard !words.isEmpty else { return displayCommand }
        var units: [String] = []
        var index = 0
        // The executable and every leading non-flag token form the first unit.
        var head: [String] = []
        while index < words.count, !words[index].hasPrefix("--") {
            head.append(words[index])
            index += 1
        }
        units.append(head.shellQuoted())
        while index < words.count {
            var unit = [words[index]]
            index += 1
            while index < words.count, !words[index].hasPrefix("--") {
                unit.append(words[index])
                index += 1
            }
            units.append(unit.shellQuoted())
        }
        guard units.count > 1 else { return units[0] }

        var lines: [String] = [units[0]]
        var current = ""
        for unit in units.dropFirst() {
            if current.isEmpty {
                current = unit
            } else if current.count + 1 + unit.count <= width {
                current += " " + unit
            } else {
                lines.append(current)
                current = unit
            }
        }
        if !current.isEmpty { lines.append(current) }
        return lines.enumerated().map { offset, line in
            let indented = offset == 0 ? line : "  " + line
            return offset == lines.count - 1 ? indented : indented + " \\"
        }.joined(separator: "\n")
    }
}
