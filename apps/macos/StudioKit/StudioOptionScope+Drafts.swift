import Foundation
import MereRunContract

// How each kind of draft meets its scope. A surface reads its scope from the argv it would launch
// before scoping, so the family is the one the run gets; the draft keeps what the scope hides,
// and the copy a run validates and launches resets it.

extension StudioScopeSource {
    /// A prompt mode's scope, read from the argv its draft builds as it stands: the model the
    /// composer sends (the template's when the draft names none), `--model-root`, and the
    /// selector flags are the ones the run carries.
    package func scope(mode: StudioMode, draft: StudioDraft) -> StudioOptionScope? {
        let templateID = StudioCommandAdapter.templateID(for: mode, draft: draft)
        guard let template = CommandCatalog.template(id: templateID), let capability = capability(for: templateID) else {
            return nil
        }
        let command = StudioCommandAdapter.commandDraft(mode: mode, draft: draft, template: template)
        return scope(capability: capability, commandLine: template.unscopedArguments(from: command))
    }

    /// A contract form's scope (a task draft, the console, the Command view), read from the argv
    /// the form builds. A form without `--model` scopes to the contract's default model.
    package func scope(capability: MereRunCommandCapability, form: StudioConsoleDraft) -> StudioOptionScope {
        scope(capability: capability, commandLine: StudioConsoleCommand.arguments(for: capability, draft: form))
    }

    package func scope(for draft: StudioTaskDraft) -> StudioOptionScope? {
        capability(for: draft.templateID).map { scope(capability: $0, form: draft.form) }
    }
}

extension StudioOptionScope {
    /// Of `flags`, the ones the scope takes out of a run: those the family does not use, and
    /// those it uses whose value in the command line its rule turns away.
    package func withheld(_ flags: some Sequence<String>) -> Set<String> {
        let unused = unusedFlags
        return Set(flags.filter { !allows($0) || unused.contains($0) })
    }

    /// Why the model cannot run the command at all: it is excluded from it, or its selectors
    /// match no family. The readiness card blocks the run with this sentence.
    package var blockingReason: String? {
        switch resolution {
        case .excluded, .unmatched: return refusal
        case .unrouted, .family, .unidentified: return nil
        }
    }
}

// MARK: - Prompt modes

extension StudioDraft {
    /// A mode's fresh draft before this machine's recommendations: what a value the scope hides
    /// goes back to in the copy a run launches.
    package static func baseline(for mode: StudioMode) -> StudioDraft {
        var draft = StudioDraft()
        draft.reset(for: mode)
        return draft
    }

    /// The copy a run validates and launches: every binding the scope withholds goes back to its
    /// baseline, or to the value the family runs where the family takes the option but not this
    /// value. A composite editor's own fields (seconds-or-frames, "override the preset steps")
    /// go back too once every flag it writes is withheld. `self` keeps every value, so switching
    /// back to a model that uses them brings them back.
    package func scoped(to scope: StudioOptionScope, mode: StudioMode) -> StudioDraft {
        guard scope.family != nil else { return self }
        let bindings = StudioContractBindings.bindings(for: mode)
        let withheld = scope.withheld(bindings.keys)
        guard !withheld.isEmpty else { return self }
        let baseline = Self.baseline(for: mode)
        var scoped = self
        for flag in withheld {
            guard let binding = bindings[flag] else { continue }
            if let option = scope.option(flag), option.defaultValue != nil {
                binding.write(&scoped, StudioContractField(option: option, bindings: [binding]).defaultValue)
            } else {
                binding.reset(&scoped, to: baseline)
            }
        }
        for override in StudioContractOverrides.overrides(for: mode) where !override.companions.isEmpty {
            let declared = override.flags.filter { flag in scope.capability.options.contains { $0.flag == flag } }
            guard !declared.isEmpty, declared.allSatisfy(withheld.contains) else { continue }
            for companion in override.companions { companion.reset(&scoped, to: baseline) }
        }
        return scoped
    }
}

// MARK: - Contract forms

extension StudioConsoleDraft {
    /// The form a run launches: without the values the scope withholds or that repeat the
    /// family's default, so the CLI runs the family's own. `self` keeps them, so switching back
    /// to a model that uses them brings them back. Extra arguments stay as typed; the CLI's gate
    /// answers them.
    package func scoped(to scope: StudioOptionScope) -> StudioConsoleDraft {
        guard let family = scope.family else { return self }
        var scoped = self
        for flag in scope.withheld(values.keys) { scoped.values[flag] = nil }
        // The family runs its own default when the flag is left off, so the run leaves it off,
        // as `StudioOptionScopes.filtered` does for a template's argv.
        for flag in scoped.values.keys {
            guard let option = scope.option(flag),
                  let familyDefault = option.familyRules.first(where: { $0.family == family.id })?.defaultValue,
                  option.reads(scoped.text(flag), asOneOf: [familyDefault]) else { continue }
            scoped.values[flag] = nil
        }
        return scoped
    }
}

// MARK: - The note

/// What a surface says about options its model does not take, or about a model the CLI is still
/// identifying. `StudioScopeNote` draws it under the composer's chips, at the top of the
/// inspectors, and as the Command view's "Not sent" line.
package struct StudioScopeNotice: Equatable {
    package enum Kind: Equatable {
        /// `catalog resolve` is running for a model the contract does not list.
        case identifying
        /// The CLI could not say which family the model is; every option shows.
        case unidentified
        /// The family leaves out, or replaces, values the draft holds.
        case unused
    }

    package let kind: Kind
    package let title: String
    package let details: [String]

    package var accessibilityLabel: String {
        ([title] + details).joined(separator: " ")
    }

    /// The note for `scope`, given the flags a surface holds a value for that the family leaves
    /// out (`hidden`) or runs with its own value (`replaced`), each in contract order.
    package init?(scope: StudioOptionScope, hidden: [String], replaced: [String]) {
        switch scope.identity {
        case .pending(let model):
            self.init(kind: .identifying, title: "Identifying \(Self.name(model))…",
                      details: ["Every option shows until mere.run knows which model this is."])
            return
        case .failed(let model):
            self.init(kind: .unidentified, title: "mere.run couldn't identify \(Self.name(model))",
                      details: ["Every option is shown, and the CLI checks them when it runs."])
            return
        case .notNeeded:
            break
        }
        guard let family = scope.family, !(hidden.isEmpty && replaced.isEmpty) else { return nil }
        let label = { (flag: String) in scope.capability.options.first { $0.flag == flag }?.label ?? flag }
        let replacements = replaced.map { flag in
            let yours = scope.invocation.values[flag]?.joined(separator: ", ") ?? ""
            guard let runs = scope.option(flag)?.defaultValue else {
                return "\(label(flag)): \(family.title) doesn't take \(yours); it's kept for other models."
            }
            return "\(label(flag)): \(family.title) runs \(runs); your \(yours) is kept."
        }
        if hidden.isEmpty {
            self.init(kind: .unused, title: replacements[0], details: Array(replacements.dropFirst()))
        } else {
            self.init(
                kind: .unused,
                title: "Not used by \(family.title): \(hidden.map(label).joined(separator: ", ")).",
                details: replacements + ["Your values are kept for when you switch back."]
            )
        }
    }

    package init(kind: Kind, title: String, details: [String]) {
        self.kind = kind
        self.title = title
        self.details = details
    }

    /// A folder by its name; anything else as typed.
    private static func name(_ model: String) -> String {
        model.contains("/") ? URL(fileURLWithPath: model).lastPathComponent : model
    }
}

extension StudioOptionScope {
    /// A prompt mode's note: the bound fields the draft sets away from the mode's baseline that
    /// the family leaves out or replaces. A composite editor's own fields count for its flags
    /// (MiniMax-H3's steps live beside `--steps`, not in its binding).
    package func notice(mode: StudioMode, draft: StudioDraft) -> StudioScopeNotice? {
        let bindings = StudioContractBindings.bindings(for: mode)
        let baseline = StudioDraft.baseline(for: mode)
        let changed = capability.options.map(\.flag).filter { flag in
            guard let binding = bindings[flag] else { return false }
            let companions = StudioContractOverrides.override(forFlag: flag, mode: mode)?.companions ?? []
            return ([binding] + companions).contains { $0.isChanged(draft, baseline) }
        }
        return notice(changed: changed)
    }

    /// A contract form's note: the values the form holds that the family leaves out or replaces.
    /// With a `baseline` (a task's fresh draft), only values that differ from it count, so the
    /// template's own defaults for other models are not listed; without one (the console, where
    /// what is shown is what runs), every value does.
    package func notice(form: StudioConsoleDraft, baseline: StudioConsoleDraft? = nil) -> StudioScopeNotice? {
        let held = capability.options.map(\.flag).filter { flag in
            guard !form.text(flag).isEmpty else { return false }
            return baseline.map { $0[flag] != form[flag] } ?? true
        }
        return notice(changed: held)
    }

    private func notice(changed: [String]) -> StudioScopeNotice? {
        let unused = unusedFlags
        return StudioScopeNotice(
            scope: self,
            hidden: changed.filter { !allows($0) },
            replaced: changed.filter { allows($0) && unused.contains($0) }
        )
    }
}
