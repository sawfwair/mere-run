import Foundation
import MereRunContract
@testable import StudioKit
import XCTest

/// Studio equals the contract, for every routed command and every one of its runtime families.
///
/// For each (capability, family) and each Studio surface that reaches the capability — a prompt
/// mode's composer, a task draft, the Command Console — the harness fills every control the
/// surface binds with a value the family accepts (or, for an option the family does not use, any
/// value the contract accepts), selects the family through its first model (a local folder the
/// CLI identifies when it lists none) and its selector flags, and checks:
///
/// - shown: the flags the surface draws equal the flags it can bind (the same surface with the
///   command unrouted) that `options(forFamily:)` keeps;
/// - validated: the flags the validation path receives are ones the family uses, and every one
///   the surface can bind is shown;
/// - emitted: the launched argv passes the CLI gate for the family (`violations` is empty), sends
///   nothing the surface can bind but did not show, and scopes to the same family.
///
/// It covers whatever routing the contract declares, so a domain change that routes a command is
/// checked without a line here. `Fixtures/model-scope/<capability>.txt` records the three sets
/// per family and surface so a reviewer sees what a contract change does to Studio; the files are
/// generated, never edited: `./scripts/update-studio-model-scope-fixtures.sh`.
final class StudioModelScopeGoldenTests: XCTestCase {
    func testEveryRoutedFamilyShowsValidatesAndSendsWhatTheContractAllows() throws {
        let rendered = ModelScopeFixtures.render()
        let fileManager = FileManager.default
        if ProcessInfo.processInfo.environment["MERERUN_UPDATE_MODEL_SCOPE_FIXTURES"] == "1" {
            try fileManager.createDirectory(at: ModelScopeFixtures.directory, withIntermediateDirectories: true)
            for stale in try ModelScopeFixtures.committedFiles() where rendered[stale] == nil {
                try fileManager.removeItem(at: ModelScopeFixtures.url(for: stale))
            }
            for (capability, text) in rendered {
                try text.write(to: ModelScopeFixtures.url(for: capability), atomically: true, encoding: .utf8)
            }
            return
        }
        let committed = try ModelScopeFixtures.committedFiles()
        XCTAssertEqual(
            committed, Set(rendered.keys),
            "Routed commands and model-scope fixtures differ. Run ./scripts/update-studio-model-scope-fixtures.sh."
        )
        for (capability, text) in rendered.sorted(by: { $0.key < $1.key }) where committed.contains(capability) {
            let recorded = try String(contentsOf: ModelScopeFixtures.url(for: capability), encoding: .utf8)
            guard recorded != text else { continue }
            let lines = Set(recorded.split(separator: "\n")).symmetricDifference(text.split(separator: "\n"))
            XCTFail("""
                \(capability): Studio's scoped surfaces changed (\(lines.sorted().prefix(4).joined(separator: " | "))). \
                If the contract change is intended, run ./scripts/update-studio-model-scope-fixtures.sh and review the diff.
                """)
        }
    }

    func testTheHarnessReachesEveryRoutedCommandStudioRuns() {
        for capability in MereRunCapabilityCatalog.document.commands where capability.routing != nil {
            let surfaces = ModelScopeFixtures.surfaces(for: capability)
            if CommandTemplateID.allCases.contains(where: { $0.capabilityID == capability.id }) {
                XCTAssertFalse(surfaces.isEmpty, "\(capability.id) has a template but no scoped surface")
            }
        }
    }
}

enum ModelScopeFixtures {
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/model-scope")

    static func url(for capability: String) -> URL {
        directory.appendingPathComponent(capability + ".txt")
    }

    static func committedFiles() throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        return Set(try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".txt") }
            .map { String($0.dropLast(4)) })
    }

    enum Surface {
        case composer(StudioMode, StudioReadImageAction)
        case task(StudioTask, CommandTemplateID)
        case console(CommandTemplateID)

        var name: String {
            switch self {
            case .composer(let mode, let action): return mode == .readImage ? "composer.readImage.\(action.rawValue)" : "composer.\(mode.rawValue)"
            case .task(_, let templateID): return "task.\(templateID.rawValue)"
            case .console(let templateID): return "console.\(templateID.rawValue)"
            }
        }
    }

    /// The surfaces that reach `capability`: the prompt modes that run it, the task drafts whose
    /// variants include it, and the console for every template that runs it.
    static func surfaces(for capability: MereRunCommandCapability) -> [Surface] {
        var surfaces: [Surface] = []
        for mode in StudioMode.allCases {
            let actions: [StudioReadImageAction] = mode == .readImage ? StudioReadImageAction.allCases : [.inspect]
            for action in actions where StudioCommandAdapter.templateID(for: mode, draft: {
                var draft = StudioDraft()
                draft.readImageAction = action
                return draft
            }()).capabilityID == capability.id {
                surfaces.append(.composer(mode, action))
            }
        }
        var seen: Set<CommandTemplateID> = []
        for task in StudioTask.allCases where task.usesTaskDraft {
            for template in task.variantTemplates where template.id.capabilityID == capability.id && seen.insert(template.id).inserted {
                surfaces.append(.task(task, template.id))
            }
        }
        for template in CommandCatalog.templates where template.id.capabilityID == capability.id {
            surfaces.append(.console(template.id))
        }
        return surfaces
    }

    /// One fixture per routed capability: `family<TAB>surface<TAB>set<TAB>flags`, sorted.
    static func render() -> [String: String] {
        var files: [String: String] = [:]
        for capability in MereRunCapabilityCatalog.document.commands {
            guard let routing = capability.routing else { continue }
            var lines: [String] = []
            for family in routing.families {
                let target = Target(capability: capability, family: family)
                for surface in surfaces(for: capability) {
                    lines += target.record(surface)
                }
            }
            files[capability.id] = "# \(capability.id): the flags each Studio surface shows, validates, and sends per runtime family.\n"
                + "# Generated by StudioModelScopeGoldenTests; run ./scripts/update-studio-model-scope-fixtures.sh.\n"
                + lines.sorted().joined(separator: "\n") + "\n"
        }
        return files
    }

    /// One family of one capability, and the sources that scope it with and without routing.
    struct Target {
        let capability: MereRunCommandCapability
        let family: MereRunRuntimeFamily
        /// The value the family's model flag carries: its first model, or a local folder the CLI
        /// identifies as the family; nil when the family is what the command runs with no model.
        let model: String?
        let source: StudioScopeSource
        let unrouted: StudioScopeSource
        let familyFlags: Set<String>

        init(capability: MereRunCommandCapability, family: MereRunRuntimeFamily) {
            self.capability = capability
            self.family = family
            let routing = capability.routing
            let folder = "/tmp/model-scope/\(family.id)"
            if let first = family.models.first {
                model = first
            } else if routing?.defaultModels.contains(where: { $0.family == family.id }) == true {
                model = nil
            } else {
                model = folder
            }
            let identities = StudioFixedModelIdentities([folder: .identified(.family(family.id))])
            source = StudioScopeSource(identities: identities)
            unrouted = StudioScopeSource(identities: identities, capability: { id in
                MereRunCapabilityCatalog.command(id: id).map { capability in
                    MereRunCommandCapability(
                        id: capability.id, command: capability.command, title: capability.title,
                        summary: capability.summary, arguments: capability.arguments, options: capability.options,
                        output: capability.output
                    )
                }
            })
            familyFlags = Set(capability.options(forFamily: family.id).map(\.flag))
        }

        private var modelFlags: Set<String> {
            Set((capability.routing?.modelFlags ?? []) + (capability.routing?.families.compactMap(\.modelFlag) ?? []))
        }

        private var selectorFlags: Set<String> {
            Set(capability.routing?.families.flatMap(\.selectors).map(\.flag) ?? [])
        }

        func record(_ surface: Surface) -> [String] {
            let sets: (shown: Set<String>, validated: Set<String>, emitted: Set<String>)?
            switch surface {
            case let .composer(mode, action): sets = composer(mode, action: action, surface: surface.name)
            case let .task(task, templateID): sets = taskDraft(task, templateID: templateID, surface: surface.name)
            case .console(let templateID): sets = console(templateID, surface: surface.name)
            }
            let prefix = "\(family.id)\t\(surface.name)\t"
            guard let sets else { return [prefix + "unreachable"] }
            return [
                prefix + "shown\t" + sets.shown.sorted().joined(separator: " "),
                prefix + "validated\t" + sets.validated.sorted().joined(separator: " "),
                prefix + "emitted\t" + sets.emitted.sorted().joined(separator: " "),
            ]
        }

        // MARK: Prompt modes

        private func composer(
            _ mode: StudioMode,
            action: StudioReadImageAction,
            surface: String
        ) -> (Set<String>, Set<String>, Set<String>)? {
            var draft = StudioDraft.baseline(for: mode)
            draft.readImageAction = action
            draft.prompt = "model scope"
            draft.model = model ?? ""
            let bindings = StudioContractBindings.bindings(for: mode)
            let narrowed = Dictionary(uniqueKeysWithValues: capability.options(forFamily: family.id).map { ($0.flag, $0) })
            for option in capability.options where option.flag != "--model" && !modelFlags.contains(option.flag) {
                guard let binding = bindings[option.flag] else { continue }
                let chosen = narrowed[option.flag] ?? option
                let field = StudioContractField(option: chosen, bindings: [binding])
                field.write(ModelScopeFixtures.value(for: chosen, current: field.value(in: draft)), to: &draft)
            }
            for override in StudioContractOverrides.overrides(for: mode) {
                for companion in override.companions {
                    companion.write(&draft, ModelScopeFixtures.changed(companion.read(draft)))
                }
            }
            for condition in family.selectors {
                guard let binding = bindings[condition.flag] else { return nil }
                binding.write(&draft, condition.values?.first.map { StudioContractValue.text($0) } ?? .flag(true))
            }
            guard let scope = source.scope(mode: mode, draft: draft), scope.family?.id == family.id else { return nil }

            let shown = composerShown(mode, draft: draft, source: source)
            let bindable = composerShown(mode, draft: draft, source: unrouted)
            check(shown == bindable.intersection(familyFlags), surface, "shows \(shown.subtracting(familyFlags).sorted()) the family does not use, or hides \(bindable.intersection(familyFlags).subtracting(shown).sorted())")

            let scoped = draft.scoped(to: scope, mode: mode)
            let baseline = StudioDraft.baseline(for: mode)
            let validated = Set(bindings.filter { flag, binding in
                capability.options.contains { $0.flag == flag } && binding.isChanged(scoped, baseline)
            }.keys)
            check(validated.subtracting(shown).intersection(bindable).isEmpty, surface,
                  "validates hidden \(validated.subtracting(shown).intersection(bindable).sorted())")

            guard let request = try? StudioCommandAdapter.makeRequest(mode: mode, draft: draft, validating: false, source: source) else {
                check(false, surface, "builds no request")
                return nil
            }
            let argv = request.template.arguments(from: request.draft, source: source)
            let emitted = checkEmitted(argv, bindable: bindable, shown: shown, surface: surface)
            return (shown, validated, emitted)
        }

        private func composerShown(_ mode: StudioMode, draft: StudioDraft, source: StudioScopeSource) -> Set<String> {
            let declared = Set(capability.options.map(\.flag))
            var flags = Set(StudioContractSchema.inspectorFields(for: mode, draft: draft, source: source).map(\.flag))
            flags.formUnion(StudioContractSchema.boundFields(for: mode, draft: draft, source: source).map(\.flag))
            flags.formUnion(mode.attachmentSlots(for: draft, source: source).compactMap(mode.attachmentFlag(for:)))
            flags.formUnion(mode.composerChips(for: draft, source: source).flatMap { $0.kind.flags(for: mode) })
            return flags.intersection(declared)
        }

        // MARK: Contract forms

        private func filledForm(_ form: StudioConsoleDraft) -> StudioConsoleDraft {
            var form = form
            for (index, argument) in capability.arguments.enumerated() {
                while form.arguments.count <= index { form.arguments.append("") }
                form.arguments[index] = [.file, .directory].contains(argument.kind) ? "/tmp/model-scope/\(argument.name)" : "model scope"
            }
            let narrowed = Dictionary(uniqueKeysWithValues: capability.options(forFamily: family.id).map { ($0.flag, $0) })
            for option in capability.options where !modelFlags.contains(option.flag) && !selectorFlags.contains(option.flag) {
                let chosen = narrowed[option.flag] ?? option
                form[option.flag] = ModelScopeFixtures.value(for: chosen, current: form[option.flag])
            }
            for flag in modelFlags { form.values[flag] = nil }
            if let model {
                let flag = family.modelFlag ?? (capability.routing?.modelFlags.contains("--model") == true ? "--model" : capability.routing?.modelFlags.first)
                if let flag { form[flag] = .text(model) }
            }
            for flag in selectorFlags { form.values[flag] = nil }
            for condition in family.selectors {
                form[condition.flag] = condition.values?.first.map { StudioContractValue.text($0) } ?? .flag(true)
            }
            return form
        }

        private func taskDraft(
            _ task: StudioTask,
            templateID: CommandTemplateID,
            surface: String
        ) -> (Set<String>, Set<String>, Set<String>)? {
            var draft = StudioTaskDraft(templateID: templateID)
            draft.form = filledForm(draft.form)
            guard let scope = source.scope(for: draft), scope.family?.id == family.id else { return nil }
            let shown = taskShown(task, draft: draft, source: source)
            let bindable = taskShown(task, draft: draft, source: unrouted)
            check(shown == bindable.intersection(familyFlags), surface, "shows \(shown.subtracting(familyFlags).sorted()) the family does not use, or hides \(bindable.intersection(familyFlags).subtracting(shown).sorted())")
            let validated = carried(draft.form.scoped(to: scope))
            check(validated.isSubset(of: familyFlags), surface, "validates \(validated.subtracting(familyFlags).sorted())")
            let emitted = checkEmitted(draft.run(source: source)?.arguments ?? [], bindable: bindable, shown: shown, surface: surface)
            check(emitted == validated, surface, "sends \(emitted.symmetricDifference(validated).sorted()) unlike the form it validates")
            return (shown, validated, emitted)
        }

        private func taskShown(_ task: StudioTask, draft: StudioTaskDraft, source: StudioScopeSource) -> Set<String> {
            var flags = Set(StudioTaskSchema.fields(for: task, draft: draft, source: source).flatMap(\.draftFieldIDs))
            for slot in draft.slots(source: source) {
                if case .flag(let flag) = slot.storage { flags.insert(flag) }
                if case .flagList(let flag) = slot.storage { flags.insert(flag) }
            }
            return flags.intersection(capability.options.map(\.flag))
        }

        private func console(_ templateID: CommandTemplateID, surface: String) -> (Set<String>, Set<String>, Set<String>)? {
            guard let template = CommandCatalog.template(id: templateID) else { return nil }
            let form = filledForm(StudioConsoleCommand.seed(template: template, draft: template.defaultDraft()))
            let scope = source.scope(capability: capability, form: form)
            guard scope.family?.id == family.id else { return nil }
            let shown = Set(StudioConsoleCommand.groups(for: capability, scope: scope).flatMap(\.fields).map(\.flag))
                .intersection(capability.options.map(\.flag))
            let bindable = Set(capability.options.map(\.flag))
            check(shown == familyFlags, surface, "shows \(shown.symmetricDifference(familyFlags).sorted()) unlike the family")
            let validated = carried(form.scoped(to: scope))
            check(validated.isSubset(of: familyFlags), surface, "validates \(validated.subtracting(familyFlags).sorted())")
            let run = StudioConsoleRun(template: template, draft: form, seed: template.defaultDraft(), source: source)
            let emitted = checkEmitted(run?.arguments ?? [], bindable: bindable, shown: shown, surface: surface)
            check(emitted == validated, surface, "sends \(emitted.symmetricDifference(validated).sorted()) unlike the form it validates")
            return (shown, validated, emitted)
        }

        private func carried(_ form: StudioConsoleDraft) -> Set<String> {
            Set(capability.options.map(\.flag).filter { form[$0].flag == true || (form[$0].flag == nil && !form.text($0).isEmpty) })
        }

        // MARK: Checks

        /// The declared flags `argv` sends, after checking it passes the family's gate and sends
        /// nothing the surface could have shown but did not.
        private func checkEmitted(_ argv: [String], bindable: Set<String>, shown: Set<String>, surface: String) -> Set<String> {
            let arguments = argv.starts(with: capability.command) ? Array(argv.dropFirst(capability.command.count)) : argv
            let emitted = Set(StudioArgvToken.read(arguments, capability: capability).compactMap { token -> String? in
                if case .option(let flag, _, _) = token { return flag }
                return nil
            })
            let invocation = MereRunCommandInvocation(capability: capability, arguments: arguments)
            let violations = capability.violations(invocation, family: family.id)
                .filter { $0.kind != .missingRequired || bindable.contains($0.flag) }
            check(violations.isEmpty, surface, "sends what the gate answers: \(violations.map(\.message))")
            check(emitted.intersection(bindable).isSubset(of: shown), surface,
                  "sends \(emitted.intersection(bindable).subtracting(shown).sorted()) it does not show")
            check(source.scope(capability: capability, commandLine: argv).family?.id == family.id, surface,
                  "launches a command that scopes to another family")
            return emitted
        }

        private func check(_ condition: Bool, _ surface: String, _ message: @autoclosure () -> String) {
            if !condition { XCTFail("\(capability.id) \(family.id) \(surface) \(message())") }
        }
    }

    /// A value `option` accepts that differs from `current` where it can: the family's own value
    /// when its rule fixes or lists them, a number inside its range, a choice other than the
    /// default, a path under /tmp for a file.
    static func value(for option: MereRunCapabilityOption, current: StudioContractValue) -> StudioContractValue {
        let rule = option.familyRules.first
        switch option.kind {
        case .boolean:
            return .flag(true)
        case .choice:
            let choices = rule?.values ?? option.choices
            let other = choices.first { $0 != option.defaultValue && .text($0) != current } ?? choices.first ?? ""
            return .text(other)
        case .integer, .number:
            if let values = rule?.values, let first = values.first {
                return option.kind == .integer ? .integer(Int(first) ?? 0) : .number(Double(first) ?? 0)
            }
            let range = rule?.range ?? option.range
            let base = current.numericValue ?? option.defaultValue.flatMap(Double.init) ?? range?.min ?? 0
            var candidate = base + (range?.step ?? 1)
            if let maximum = range?.max, candidate > maximum { candidate = max(range?.min ?? maximum - 1, base - (range?.step ?? 1)) }
            if let minimum = range?.min, candidate < minimum { candidate = minimum }
            return option.kind == .integer ? .integer(Int(candidate.rounded())) : .number(candidate)
        case .string:
            return .text(rule?.values?.first ?? "model scope")
        case .file, .directory:
            return .text("/tmp/model-scope/\(option.flag.drop { $0 == "-" })")
        }
    }

    /// A companion field moved off its value, keeping its type.
    static func changed(_ value: StudioContractValue) -> StudioContractValue {
        switch value {
        case .flag(let on): return .flag(!on)
        case .integer(let number): return .integer(number + 1)
        case .number(let number): return .number(number + 0.5)
        case .text, .unset: return value
        }
    }
}
