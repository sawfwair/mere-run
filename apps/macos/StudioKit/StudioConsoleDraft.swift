import Foundation
import MereRunContract

// The Command Console edits a capability, not a template. Where a designed task binds each
// contract option to a typed `StudioDraft` property, the console keeps one entry per flag and
// builds the argv from `MereRunCapabilityCatalog` itself: the command path, the positional
// arguments, then every option that carries a value, in the order the contract declares them.
//
// A console draft is seeded by reading the argv the template already produces for a
// `CommandDraft` (`StudioConsoleDraft.seed`), so opening a template — or a Library row's saved
// command — starts at exactly the command the app would have run, and `StudioConsoleDraftTests`
// holds that identity for every template in the catalog.

/// Every option of one capability, keyed by the flag the contract declares, plus its positional
/// arguments and anything typed by hand.
package struct StudioConsoleDraft: Equatable {
    /// Positional values, in the order `MereRunCommandCapability.arguments` declares them.
    package var arguments: [String] = []
    /// One entry per flag the form carries a value for. A flag with no entry is not emitted.
    package var values: [String: StudioContractValue] = [:]
    /// Arguments the contract does not describe, split as a shell would and appended verbatim.
    package var extraArguments = ""

    package init(
        arguments: [String] = [],
        values: [String: StudioContractValue] = [:],
        extraArguments: String = ""
    ) {
        self.arguments = arguments
        self.values = values
        self.extraArguments = extraArguments
    }

    package subscript(flag: String) -> StudioContractValue {
        get { values[flag] ?? .unset }
        set { values[flag] = newValue }
    }

    /// The text a flag carries, empty when it carries nothing.
    package func text(_ flag: String) -> String {
        switch self[flag] {
        case .text(let text): return text
        case .integer(let value): return String(value)
        case .number(let value): return StudioComposerPresets.decimalText(value)
        case .flag(let on): return on ? "true" : ""
        case .unset: return ""
        }
    }
}

extension StudioContractBinding where Draft == StudioConsoleDraft {
    /// One flag's entry. The flag doubles as the field id: the console has no typed draft
    /// property to name it after.
    package static func flag(_ flag: String) -> Self {
        Self(
            fieldID: flag,
            read: { $0[flag] },
            write: { draft, value in draft[flag] = value }
        )
    }

    /// One positional argument, by the index the contract declares it at.
    package static func argument(_ index: Int) -> Self {
        Self(
            fieldID: "argument.\(index)",
            read: { draft in
                guard index < draft.arguments.count else { return .unset }
                let value = draft.arguments[index]
                return value.isEmpty ? .unset : .text(value)
            },
            write: { draft, value in
                while draft.arguments.count <= index { draft.arguments.append("") }
                draft.arguments[index] = value.text ?? ""
            }
        )
    }
}

/// The console's form and argv, both read from `MereRunCapabilityCatalog`.
package enum StudioConsoleCommand {
    /// A capability's positional arguments as contract fields, so the form draws them with the
    /// same controls it draws options with. They file under their own eyebrow, ahead of the
    /// contract's groups, the way the Command view lists them.
    package static func argumentFields(for capability: MereRunCommandCapability) -> [StudioContractField<StudioConsoleDraft>] {
        capability.arguments.enumerated().map { index, argument in
            StudioContractField(
                option: MereRunCapabilityOption(
                    flag: argument.name,
                    label: argument.label,
                    kind: argument.kind,
                    required: argument.required
                ),
                bindings: [.argument(index)]
            )
        }
    }

    /// Every option the capability declares, in contract order. Nothing is filtered: the console
    /// is the surface that must reach an option no designed task has a control for.
    package static func optionFields(for capability: MereRunCommandCapability) -> [StudioContractField<StudioConsoleDraft>] {
        capability.options.map { StudioContractField(option: $0, bindings: [.flag($0.flag)]) }
    }

    /// The rows of one eyebrow group, in the order both the console and the Command view show
    /// them: positionals first, then the contract's own groups.
    package static func groups(for capability: MereRunCommandCapability) -> [StudioConsoleGroup] {
        var groups: [StudioConsoleGroup] = []
        let arguments = argumentFields(for: capability)
        if !arguments.isEmpty {
            groups.append(StudioConsoleGroup(group: .arguments, fields: arguments))
        }
        let options = optionFields(for: capability)
        for group in StudioContractGroup.allCases {
            let fields = options.filter { $0.group == group }
            guard !fields.isEmpty else { continue }
            groups.append(StudioConsoleGroup(group: .contract(group), fields: fields))
        }
        return groups
    }

    /// Whether each flag carries a value, and what it in turn depends on, so `ContractForm`
    /// can hide a row whose `depends_on` option is empty.
    package static func dependencies(
        for capability: MereRunCommandCapability,
        draft: StudioConsoleDraft
    ) -> [String: (carries: Bool, dependsOn: String?)] {
        var entries: [String: (carries: Bool, dependsOn: String?)] = [:]
        for option in capability.options {
            entries[option.flag] = (carries(option, in: draft), option.dependsOn)
        }
        return entries
    }

    /// What the console will launch: the command path, the positional arguments the draft fills,
    /// every option that carries a value in the contract's own order, then anything typed into
    /// Extra arguments.
    ///
    /// A value is emitted exactly when the draft holds one, rather than when it differs from the
    /// contract's declared default: the console is the raw surface, and what it shows is what it
    /// runs. A repeatable option emits once per non-empty line.
    package static func arguments(
        for capability: MereRunCommandCapability,
        draft: StudioConsoleDraft,
        secretFlags: Set<String> = []
    ) -> [String] {
        var argv = capability.command
        for (index, _) in capability.arguments.enumerated() {
            guard index < draft.arguments.count else { break }
            let value = draft.arguments[index].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty else { continue }
            argv.append(value)
        }
        for option in capability.options {
            // A secret never reaches argv: `commandDraft` puts it in the environment instead.
            guard !secretFlags.contains(option.flag), carries(option, in: draft) else { continue }
            if option.kind == .boolean {
                argv.append(option.flag)
                continue
            }
            for value in values(option, in: draft) {
                argv += [option.flag, value]
            }
        }
        argv += ShellWords.split(draft.extraArguments)
        return argv
    }

    /// The reason the command cannot run yet, in the contract's own words: a required positional
    /// or a required option with nothing in it.
    package static func validationMessage(
        for capability: MereRunCommandCapability,
        draft: StudioConsoleDraft
    ) -> String? {
        for (index, argument) in capability.arguments.enumerated() where argument.required {
            let value = index < draft.arguments.count ? draft.arguments[index] : ""
            if value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "\(argument.label) is required."
            }
        }
        for option in capability.options where option.required && option.kind != .boolean {
            if !carries(option, in: draft) {
                return "\(option.label) (\(option.flag)) is required."
            }
        }
        return nil
    }

    /// Where the run will write, when the contract names the option that says so. `JobStore`
    /// creates the enclosing folder from this and `ArtifactResolver` adopts the file.
    package static func outputPath(for capability: MereRunCommandCapability, draft: StudioConsoleDraft) -> String {
        guard let flag = capability.output.flag else { return "" }
        return values(capability.options.first { $0.flag == flag }, in: draft).first ?? ""
    }

    /// The draft the console starts from: the argv `template` already builds for `draft`, read
    /// back into contract values. Opening a template, or a Library row's saved command, therefore
    /// starts at exactly the command the app would have run.
    package static func seed(template: CommandTemplate, draft: CommandDraft) -> StudioConsoleDraft {
        guard let capability = template.id.capability else {
            return StudioConsoleDraft(extraArguments: draft.extraArguments)
        }
        return seed(capability: capability, arguments: template.arguments(from: draft))
    }

    /// The same reading, from argv the caller already has: a Library row records the exact
    /// arguments its run launched, so "Edit command" reopens the console on that command rather
    /// than on the draft it was built from.
    package static func seed(capability: MereRunCommandCapability, arguments argv: [String]) -> StudioConsoleDraft {
        let parsed = StudioCommandRows.parse(arguments: argv, commandPathCount: capability.command.count)
        var console = StudioConsoleDraft()
        console.arguments = capability.arguments.enumerated().map { index, _ in
            index < parsed.positional.count ? parsed.positional[index] : ""
        }
        // Positionals the contract does not declare would otherwise be dropped; keep them where
        // the user can see and edit them.
        if parsed.positional.count > capability.arguments.count {
            console.extraArguments = parsed.positional
                .dropFirst(capability.arguments.count)
                .joined(separator: " ")
        }
        let declared = Dictionary(
            capability.options.map { ($0.flag, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        var repeated: [String: [String]] = [:]
        for (flag, value) in parsed.flags {
            guard let option = declared[flag] else { continue }
            if option.kind == .boolean {
                console[flag] = .flag(true)
            } else if let value {
                repeated[flag, default: []].append(value)
            }
        }
        // Values stay the text the argv carried. Re-reading `--learning-rate 0.0001` as a number
        // and formatting it back would change the command the console shows.
        for (flag, values) in repeated {
            console[flag] = .text(values.joined(separator: "\n"))
        }
        return console
    }

    /// The `CommandDraft` the job lifecycle reads alongside the console's argv: the seed with the
    /// destination the console's own form names, so the output folder is prepared and the written
    /// file is adopted as the run's artifact.
    package static func commandDraft(
        seed: CommandDraft,
        template: CommandTemplate,
        capability: MereRunCommandCapability,
        draft: StudioConsoleDraft
    ) -> CommandDraft {
        var command = seed
        command.outputPath = outputPath(for: capability, draft: draft)
        for (flag, keyPath) in CommandLaunchEnvironment.secretFlags(for: template.id) {
            command[keyPath: keyPath] = draft.text(flag)
        }
        return command
    }

    // MARK: Values

    /// Whether the draft gives `option` something the argv would carry.
    private static func carries(_ option: MereRunCapabilityOption, in draft: StudioConsoleDraft) -> Bool {
        if option.kind == .boolean { return draft[option.flag].flag == true }
        return !values(option, in: draft).isEmpty
    }

    /// The values one option contributes: one for a plain option, one per non-empty line for a
    /// repeatable one.
    private static func values(_ option: MereRunCapabilityOption?, in draft: StudioConsoleDraft) -> [String] {
        guard let option else { return [] }
        let text = draft.text(option.flag)
        guard !text.isEmpty else { return [] }
        guard option.repeatable else {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? [] : [trimmed]
        }
        return text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

/// What the console will launch for one template: the argv, the `CommandDraft` the job
/// lifecycle reads beside it (the destination, and any secret that travels in the environment),
/// and the reason it cannot run yet.
///
/// A template the contract describes builds its argv from `MereRunCapabilityCatalog`. The Custom
/// row has no capability and keeps the catalog's own raw-argument path, which is the one editor
/// the console still writes by hand. A template that hands off to another product has nothing to
/// launch, and initializes to nil.
package struct StudioConsoleRun {
    package let arguments: [String]
    package let commandDraft: CommandDraft
    package let validationMessage: String?

    package init?(template: CommandTemplate, draft: StudioConsoleDraft, seed: CommandDraft) {
        guard template.externalURL == nil else { return nil }
        guard let capability = template.id.capability else {
            var command = seed
            command.extraArguments = draft.extraArguments
            arguments = template.arguments(from: command)
            commandDraft = command
            validationMessage = template.validationMessage(for: command)
            return
        }
        let secrets = Set(CommandLaunchEnvironment.secretFlags(for: template.id).keys)
        arguments = StudioConsoleCommand.arguments(for: capability, draft: draft, secretFlags: secrets)
        commandDraft = StudioConsoleCommand.commandDraft(
            seed: seed,
            template: template,
            capability: capability,
            draft: draft
        )
        validationMessage = StudioConsoleCommand.validationMessage(for: capability, draft: draft)
    }
}

/// One eyebrow of the console's form.
package struct StudioConsoleGroup: Identifiable {
    package let group: StudioCommandRowGroup
    package let fields: [StudioContractField<StudioConsoleDraft>]

    package var id: StudioCommandRowGroup { group }
    package var title: String { group.title }
}
