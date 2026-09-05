import Foundation

/// Editable Command values plus the simplified form they were based on. Later composer edits
/// replace only the fields they changed, preserving options available only in Command.
package struct StudioTaskCommandState: Codable, Equatable {
    package var templateID: CommandTemplateID
    package var sourceArguments: [String]
    package var form: StudioConsoleDraft

    package init(templateID: CommandTemplateID, sourceArguments: [String], form: StudioConsoleDraft) {
        self.templateID = templateID
        self.sourceArguments = sourceArguments
        self.form = form
    }

    package func resolved(source: [String]) -> StudioConsoleDraft {
        guard let capability = templateID.capability else { return form }
        let before = StudioConsoleCommand.seed(capability: capability, arguments: sourceArguments)
        let after = StudioConsoleCommand.seed(capability: capability, arguments: source)
        var merged = form
        for flag in Set(before.values.keys).union(after.values.keys) where before[flag] != after[flag] {
            merged[flag] = after[flag]
        }
        if before.arguments != after.arguments { merged.arguments = after.arguments }
        if before.extraArguments != after.extraArguments { merged.extraArguments = after.extraArguments }
        return merged
    }

    package var withoutSecrets: StudioTaskCommandState {
        var saved = self
        for flag in Set(CommandLaunchEnvironment.secretFlags(for: templateID).keys)
            .union(["--api-key", "--infinity-api-key", "--admin-password", "--hf-token"]) {
            saved.form.values[flag] = nil
        }
        saved.form.extraArguments = ShellWords.split(saved.form.extraArguments).maskingSecrets().shellQuoted()
        saved.sourceArguments = saved.sourceArguments.maskingSecrets()
        return saved
    }
}

extension StudioConsoleDraft {
    package func applyingChanges(from previous: StudioConsoleDraft, to draft: inout StudioDraft,
                                 mode: StudioMode, templateID: CommandTemplateID) {
        guard let capability = templateID.capability else { return }
        if arguments != previous.arguments,
           let positional = capability.arguments.first, ["prompt", "caption", "text"].contains(positional.name) {
            draft.prompt = arguments.first ?? ""
        }
        let bindings = StudioContractBindings.bindings(for: mode)
        for option in capability.options where self[option.flag] != previous[option.flag] {
            let text = text(option.flag)
            let value: StudioContractValue
            switch option.kind {
            case .integer: value = Int(text).map(StudioContractValue.integer) ?? .unset
            case .number: value = Double(text).map(StudioContractValue.number) ?? .unset
            case .boolean: value = .flag(self[option.flag].flag == true)
            default: value = .text(text)
            }
            bindings[option.flag]?.write(&draft, value)
            if ["--prompt", "--text", "--query"].contains(option.flag) { draft.prompt = text }
        }
    }
}
