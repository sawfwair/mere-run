import Foundation
import Testing

@testable import MereRunContract

// MARK: - Every routed capability in the catalog is well formed

private let routed = MereRunCapabilityCatalog.document.commands.compactMap { capability in
    capability.routing.map { (capability, $0) }
}

@Test func routedCapabilitiesDeclareUniqueFamiliesAndKnownFlags() {
    #expect(!routed.isEmpty)
    for (capability, routing) in routed {
        let id = capability.id
        let familyIDs = routing.families.map(\.id)
        let flags = Set(capability.options.map(\.flag))
        #expect(!familyIDs.isEmpty, "\(id): routing declares no family")
        #expect(Set(familyIDs).count == familyIDs.count, "\(id): duplicate family ids")
        #expect(familyIDs.allSatisfy { $0.range(of: #"^[a-z0-9]+(-[a-z0-9.]+)*$"#, options: .regularExpression) != nil },
                "\(id): family ids are kebab-case")
        #expect(Set(routing.modelFlags).isSubset(of: flags), "\(id): model flags must be declared options")
        for family in routing.families {
            if let modelFlag = family.modelFlag {
                #expect(flags.contains(modelFlag), "\(id) \(family.id): model flag \(modelFlag) is not declared")
            }
            for selector in family.selectors {
                expectValid(selector, in: capability, context: "\(id) \(family.id) selector")
            }
        }
        for rule in routing.defaultModels {
            for condition in rule.whenAny {
                expectValid(condition, in: capability, context: "\(id) default rule")
            }
        }
        let excluded = Set(routing.excludedModels.map(\.id))
        #expect(excluded.count == routing.excludedModels.count, "\(id): duplicate excluded ids")
        #expect(routing.excludedModels.allSatisfy { !$0.reason.isEmpty }, "\(id): excluded models need a reason")
        let listed = Set(routing.families.flatMap(\.models))
        #expect(listed.isDisjoint(with: excluded), "\(id): \(listed.intersection(excluded)) are both listed and excluded")
        let identified = Set(routing.identifiedModels)
        #expect(identified.count == routing.identifiedModels.count, "\(id): duplicate identified ids")
        #expect(identified.isDisjoint(with: listed.union(excluded)),
                "\(id): \(identified.intersection(listed.union(excluded))) are identified and also listed or excluded")
        #expect(identified.isEmpty || !routing.modelFlags.isEmpty, "\(id): identified models need a model flag")
    }
}

@Test func noModelBelongsToTwoFamiliesWhoseSelectorsCanBothMatch() {
    for (capability, routing) in routed {
        for (index, first) in routing.families.enumerated() {
            for second in routing.families.dropFirst(index + 1) {
                let shared = Set(first.models).intersection(second.models)
                guard !shared.isEmpty || routing.routesBySelectors else { continue }
                let disjoint = first.selectors.contains { left in second.selectors.contains(where: left.excludes) }
                #expect(disjoint, "\(capability.id): \(first.id) and \(second.id) can both match \(shared.sorted())")
            }
        }
    }
}

@Test func everyFamilyIsReachable() {
    for (capability, routing) in routed {
        // A family reached only through an identified model is checked by the CLI's identifier
        // tests, which know which checkpoints each identified model can land on.
        for family in routing.families where family.models.isEmpty && !routing.routesBySelectors
            && routing.identifiedModels.isEmpty {
            #expect(
                routing.defaultModels.contains { $0.family == family.id },
                "\(capability.id) \(family.id) lists no model, so a default rule has to name it"
            )
        }
    }
}

/// A macOS default rule names one family, or lists candidates the CLI picks between by machine.
/// Candidates that span families resolve through the caller's `chooseDefault`; Core's
/// `ModelFamilyIdentifier` must register a chooser for them (`ManagedModelFamilyCoverageTests`).
@Test func everyDefaultRuleResolvesToOneFamilyOnMacOS() {
    for (capability, routing) in routed {
        for rule in routing.defaultModels where rule.applies(on: "macos") {
            if rule.family == nil, rule.models.count == 1, routing.identifiedModels.contains(rule.models[0]) {
                continue  // the identifier answers it, as for an explicit --model
            }
            let families = rule.family.map { [$0] }
                ?? Array(Set(rule.models.flatMap { model in routing.families.filter { $0.models.contains(model) }.map(\.id) }))
            let machineChosen = rule.family == nil && rule.models.count > 1
                && rule.models.allSatisfy { model in routing.families.contains { $0.models.contains(model) } }
            #expect(families.count == 1 || machineChosen && families.count > 1,
                    "\(capability.id): default \(rule.models) resolves to \(families)")
            #expect(families.allSatisfy { routing.family(id: $0) != nil }, "\(capability.id): default names an unknown family")
        }
        let blank = MereRunCommandInvocation(capability: capability, arguments: [])
        let chosen = capability.resolveFamily(blank, chooseDefault: { $0.last })
        guard case .family = chosen else {
            Issue.record("\(capability.id): a blank command line must resolve to a family on macOS, got \(chosen)")
            continue
        }
    }
}

@Test func aDefaultThatSpansFamiliesResolvesThroughTheMachineChooser() {
    enum TierFamily: String, MereRunFamilyID { case small, large }
    let tiers = MereRunCommandCapability(
        id: "tier.run", command: ["tier", "run"], title: "Run", summary: "A test capability.",
        options: [MereRunCapabilityOption(flag: "--model", label: "Model", kind: .string)],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: ["--model"],
            defaultModels: [.always("tier-small", "tier-large")],
            families: [
                .init(TierFamily.small, title: "Small", models: ["tier-small"]),
                .init(TierFamily.large, title: "Large", models: ["tier-large"])
            ]
        )
    )
    let blank = MereRunCommandInvocation(capability: tiers, arguments: [])
    #expect(tiers.resolveFamily(blank) == .unidentified(model: "tier-small, tier-large"))
    #expect(tiers.resolveFamily(blank, chooseDefault: { _ in "tier-large" }) == .family(id: "large", model: "tier-large", source: .defaultModel))
    #expect(tiers.resolveFamily(blank, chooseDefault: { _ in "tier-other" }) == .unidentified(model: "tier-small, tier-large"),
            "a choice outside the candidates is not trusted")
    #expect(tiers.resolutionReport(blank, chooseDefault: { _ in "tier-small" }).family == "small")
}

@Test func optionScopesNameDeclaredFamiliesAndValidValues() {
    for (capability, routing) in routed {
        let familyIDs = Set(routing.families.map(\.id))
        for option in capability.options {
            let context = "\(capability.id) \(option.flag)"
            let used = Set(option.families ?? Array(familyIDs))
            #expect(used.isSubset(of: familyIDs), "\(context): families \(used.subtracting(familyIDs))")
            #expect(Set(option.ignoredBy).isSubset(of: familyIDs), "\(context): ignored_by names an unknown family")
            #expect(used.isDisjoint(with: option.ignoredBy), "\(context): a family both uses and ignores it")
            #expect(Set(option.familyRules.map(\.family)).count == option.familyRules.count, "\(context): one rule per family")
            for rule in option.familyRules {
                let ruleContext = "\(context) rule \(rule.family)"
                if option.ignoredBy.contains(rule.family) {
                    #expect(rule.values != nil || rule.range != nil, "\(ruleContext): name the values the family tolerates")
                    #expect(rule.defaultValue == nil && !rule.required && rule.maxCount == nil && rule.severity == .error,
                            "\(ruleContext): a rule on an ignoring family only lists tolerated values")
                } else {
                    #expect(used.contains(rule.family), "\(ruleContext): the family must use or ignore the option")
                }
                #expect(option.kind != .boolean, "\(ruleContext): Booleans take no rules")
                for value in (rule.values ?? []) + [rule.defaultValue].compactMap({ $0 }) {
                    #expect(parses(value, as: option), "\(ruleContext): \(value) is not a valid \(option.kind)")
                    #expect(within(value, option.range), "\(ruleContext): \(value) is outside the option's range")
                }
                if rule.values?.count == 1 {
                    #expect(rule.defaultValue == nil, "\(ruleContext): a single allowed value is already the default")
                }
                if let range = rule.range, let min = range.min, let max = range.max {
                    #expect(min <= max, "\(ruleContext): empty range")
                }
                if rule.maxCount != nil {
                    #expect(option.repeatable, "\(ruleContext): max_count needs a repeatable option")
                }
            }
        }
    }
}

@Test func choiceSpellingsNameDeclaredChoices() {
    for capability in MereRunCapabilityCatalog.document.commands {
        for option in capability.options {
            guard let spellings = option.choiceSpellings else { continue }
            let context = "\(capability.id) \(option.flag)"
            #expect(option.kind == .choice, "\(context): only a choice has other spellings")
            for (alias, choice) in spellings.aliases {
                #expect(option.choices.contains(choice), "\(context): \(alias) names \(choice), which is not a choice")
                #expect(!option.choices.contains(alias), "\(context): \(alias) is already a choice")
                if spellings.ignoresCase {
                    #expect(alias == alias.lowercased(), "\(context): case-insensitive aliases are written lowercased")
                }
            }
        }
    }
}

@Test func spellingsAreUniqueWithinACapability() {
    for capability in MereRunCapabilityCatalog.document.commands {
        let spellings = capability.options.flatMap(\.spellings)
        #expect(Set(spellings).count == spellings.count, "\(capability.id): two options share a spelling")
    }
}

@Test func unroutedCapabilitiesCarryNoOptionScopes() {
    for capability in MereRunCapabilityCatalog.document.commands where capability.routing == nil {
        #expect(
            capability.options.allSatisfy { $0.families == nil && $0.ignoredBy.isEmpty && $0.familyRules.isEmpty },
            "\(capability.id) scopes options by family but declares no routing"
        )
    }
}

@Test func excludedModelsFailBeforeAnythingElse() {
    for (capability, routing) in routed {
        for excluded in routing.excludedModels {
            guard let flag = routing.modelFlags.last else { continue }
            let invocation = MereRunCommandInvocation(capability: capability, arguments: [flag, excluded.id])
            #expect(capability.resolveFamily(invocation) == .excluded(excluded), "\(capability.id) \(excluded.id)")
            let report = capability.resolutionReport(invocation)
            #expect(report.violations == ["\(excluded.id) can't run \(capability.command.joined(separator: " ")): \(excluded.reason)"])
        }
    }
}

private func expectValid(_ condition: MereRunFlagCondition, in capability: MereRunCommandCapability, context: String) {
    guard let option = capability.options.first(where: { $0.flag == condition.flag }) else {
        Issue.record("\(context): \(condition.flag) is not a declared option")
        return
    }
    for value in condition.values ?? [] {
        #expect(parses(value, as: option), "\(context): \(value) is not a valid \(condition.flag)")
    }
}

private func parses(_ value: String, as option: MereRunCapabilityOption) -> Bool {
    switch option.kind {
    case .integer: Int(value) != nil
    case .number: Double(value) != nil
    case .choice: option.choices.contains(value)
    case .boolean: value == "true" || value == "false"
    case .string, .file, .directory: true
    }
}

private func within(_ value: String, _ range: MereRunCapabilityRange?) -> Bool {
    guard let range, let number = Double(value) else { return true }
    return (range.min.map { number >= $0 } ?? true) && (range.max.map { number <= $0 } ?? true)
}

// MARK: - The resolver, on a capability that exercises every branch

private enum ClipFamily: String, MereRunFamilyID {
    case quick
    case full
    case wide
    case remote
}

private let clipOptions: [MereRunCapabilityOption] = [
    MereRunCapabilityOption(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
    MereRunCapabilityOption(flag: "--model-root", label: "Model root", kind: .directory),
    MereRunCapabilityOption(flag: "--backend", label: "Backend", kind: .choice, choices: ["local", "remote"],
                            defaultValue: "local"),
    MereRunCapabilityOption(flag: "--hq", label: "High quality", kind: .boolean),
    MereRunCapabilityOption(flag: "--steps", aliases: ["-s"], label: "Steps", kind: .integer, defaultValue: "8",
                            range: .init(min: 1, max: 100))
        .scoped(ClipFamily.except(.remote), .rule(.quick, values: ["4"]), .rule(.full, range: .init(min: 10, max: 60))),
    MereRunCapabilityOption(flag: "--cfg", label: "CFG", kind: .number)
        .scoped(ClipFamily.only(.full, ignoredBy: [.quick])),
    MereRunCapabilityOption(flag: "--image", label: "Image", kind: .file, repeatable: true)
        .scoped(ClipFamily.only(.full, .wide), .rule(.wide, required: true), .rule(.full, maxCount: 2, severity: .warning)),
    MereRunCapabilityOption(flag: "--mode", label: "Mode", kind: .choice, choices: ["fast", "slow", "exact"])
        .scoped(ClipFamily.rule(.full, values: ["slow", "exact"], defaultValue: "slow")),
    MereRunCapabilityOption(flag: "--strength", label: "Strength", kind: .number, defaultValue: "0.5")
        .scoped(ClipFamily.only(.full, ignoredBy: [.quick]), .rule(.quick, values: ["0.5"]))
]

private let clip = MereRunCommandCapability(
    id: "clip.render",
    command: ["clip", "render"],
    title: "Render clips",
    summary: "A test capability.",
    options: clipOptions,
    output: .init(kind: .text),
    routing: MereRunCapabilityRouting(
        modelFlags: ["--model-root", "--model"],
        defaultModels: [
            .init(whenAny: [.init(flag: "--hq")], models: ["clip-full"]),
            .always("clip-quick")
        ],
        families: [
            .init(ClipFamily.quick, title: "Quick", models: ["clip-quick", "clip-shared"],
                  selectors: [.init(flag: "--backend", values: ["local"])]),
            .init(ClipFamily.full, title: "Full", models: ["clip-full"]),
            .init(ClipFamily.wide, title: "Wide", models: ["clip-wide"]),
            .init(ClipFamily.remote, title: "Remote", models: ["clip-shared"],
                  selectors: [.init(flag: "--backend", values: ["remote"])])
        ],
        excludedModels: .models(["clip-lm"], reason: "It is a language model; use `clip caption`.")
    )
)

/// A capability without routing: every option, no family.
private let plain = MereRunCommandCapability(
    id: "plain.run", command: ["plain", "run"], title: "Run", summary: "A test capability.",
    options: [MereRunCapabilityOption(flag: "--model", label: "Model", kind: .string)],
    output: .init(kind: .text)
)

private func invocation(_ arguments: String...) -> MereRunCommandInvocation {
    MereRunCommandInvocation(capability: clip, arguments: arguments)
}

@Test func theResolverFollowsTheDocumentedOrder() {
    let identify: (String) -> MereRunModelIdentification? = { model in
        switch model {
        case "Clip-Full": .managedModel("clip-full")
        case "/models/wide": .family("wide")
        case "/models/remote-looking": .family("remote")
        default: nil
        }
    }
    let cases: [(MereRunCommandInvocation, MereRunFamilyResolution)] = [
        (invocation(), .family(id: "quick", model: "clip-quick", source: .defaultModel)),
        (invocation("--hq"), .family(id: "full", model: "clip-full", source: .defaultModel)),
        (invocation("-m", "clip-full"), .family(id: "full", model: "clip-full", source: .model)),
        (invocation("--model=clip-wide"), .family(id: "wide", model: "clip-wide", source: .model)),
        (invocation("--model", "clip-full", "--model-root", "clip-wide"),
         .family(id: "wide", model: "clip-wide", source: .model)),
        (invocation("--model", "clip-shared"), .family(id: "quick", model: "clip-shared", source: .selector)),
        (invocation("--model", "clip-shared", "--backend", "remote"),
         .family(id: "remote", model: "clip-shared", source: .selector)),
        (invocation("--model", "clip-quick", "--backend", "remote"),
         .unmatched(model: "clip-quick", detail: "clip-quick can't run with these options: Quick needs --backend local.")),
        (invocation("--model", "clip-lm"),
         .excluded(.init(id: "clip-lm", reason: "It is a language model; use `clip caption`."))),
        (invocation("--model", "/models/unknown"), .unidentified(model: "/models/unknown")),
        (invocation("--model", "Clip-Full"), .family(id: "full", model: "clip-full", source: .model)),
        (invocation("--model", "/models/wide"), .family(id: "wide", model: "/models/wide", source: .identified)),
        (invocation("--model", "/models/remote-looking"),
         .unmatched(model: "/models/remote-looking",
                    detail: "/models/remote-looking can't run with these options: Remote needs --backend remote."))
    ]
    for (given, expected) in cases {
        #expect(clip.resolveFamily(given, identify: identify) == expected, "\(given.values)")
    }
    #expect(plain.resolveFamily(MereRunCommandInvocation(capability: plain, arguments: [])) == .unrouted)
}

@Test func installedModelsCanOverrideAListedFamilyOnlyWhenTheRoutingSaysSo() throws {
    // An override root holds a Full checkpoint whatever quick id or default names it.
    let identify: (String) -> MereRunModelIdentification? = { _ in .family("full") }
    let listed = invocation("--model", "clip-quick")
    #expect(clip.resolveFamily(listed, identify: identify) == .family(id: "quick", model: "clip-quick", source: .model))

    let routing = try #require(clip.routing)
    let installed = MereRunCommandCapability(
        id: clip.id, command: clip.command, title: clip.title, summary: clip.summary, options: clip.options,
        output: clip.output,
        routing: MereRunCapabilityRouting(
            modelFlags: routing.modelFlags, defaultModels: routing.defaultModels, families: routing.families,
            excludedModels: routing.excludedModels, identifiesInstalledModels: true
        )
    )
    let read = { (arguments: [String], identify: (String) -> MereRunModelIdentification?) in
        installed.resolveFamily(MereRunCommandInvocation(capability: installed, arguments: arguments), identify: identify)
    }
    #expect(read(["--model", "clip-quick"], identify) == .family(id: "full", model: "clip-quick", source: .identified))
    #expect(read([], identify) == .family(id: "full", model: "clip-quick", source: .identified))
    #expect(read(["--model", "clip-quick"], { _ in nil }) == .family(id: "quick", model: "clip-quick", source: .model))
    #expect(read([], { _ in nil }) == .family(id: "quick", model: "clip-quick", source: .defaultModel))
    #expect(read(["--model", "clip-lm"], identify)
        == .excluded(.init(id: "clip-lm", reason: "It is a language model; use `clip caption`.")))

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let json = String(decoding: try encoder.encode(try #require(installed.routing)), as: UTF8.self)
    #expect(json.contains(#""identifies_installed_models":true"#))
    #expect(try JSONDecoder().decode(MereRunCapabilityRouting.self, from: Data(json.utf8)) == installed.routing)
    #expect(!String(decoding: try encoder.encode(routing), as: UTF8.self).contains("identifies_installed_models"))
}

@Test func selectorRoutedFamiliesCheckTheirOwnModelFlag() {
    enum ReaderFamily: String, MereRunFamilyID { case lighton, infinity }
    let reader = MereRunCommandCapability(
        id: "reader.read", command: ["reader", "read"], title: "Read", summary: "A test capability.",
        options: [
            MereRunCapabilityOption(flag: "--backend", label: "Backend", kind: .choice,
                                    choices: ["lighton", "infinity"], defaultValue: "lighton"),
            MereRunCapabilityOption(flag: "--model", label: "Model", kind: .string),
            MereRunCapabilityOption(flag: "--infinity-model", label: "Infinity model", kind: .string)
        ],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: ["--model"],
            families: [
                .init(ReaderFamily.lighton, title: "LightOn", models: ["reader-lighton"],
                      selectors: [.init(flag: "--backend", values: ["lighton"])]),
                .init(ReaderFamily.infinity, title: "Infinity", models: ["reader-infinity"],
                      selectors: [.init(flag: "--backend", values: ["infinity"])], modelFlag: "--infinity-model")
            ]
        )
    )
    let read = { (arguments: [String]) in
        reader.resolveFamily(MereRunCommandInvocation(capability: reader, arguments: arguments))
    }
    #expect(read([]) == .family(id: "lighton", model: nil, source: .selector))
    #expect(read(["--backend", "infinity", "--infinity-model", "reader-infinity"])
        == .family(id: "infinity", model: "reader-infinity", source: .selector))
    #expect(read(["--model", "/local/lighton"]) == .family(id: "lighton", model: "/local/lighton", source: .selector))
    #expect(read(["--model", "reader-infinity"]) == .unmatched(
        model: "reader-infinity",
        detail: "reader-infinity runs on Infinity, not LightOn; change the model or the selector flags."
    ))
}

/// A model whose checkpoint depends on what is installed resolves only through `identify`, as an
/// explicit model and as a default; without an answer it stays unidentified for a shell to ask
/// `catalog resolve`.
@Test func identifiedModelsResolveThroughTheIdentifier() throws {
    enum ReelFamily: String, MereRunFamilyID { case draft, full }
    let reel = MereRunCommandCapability(
        id: "reel.render", command: ["reel", "render"], title: "Reel", summary: "A test capability.",
        options: [
            MereRunCapabilityOption(flag: "--model", label: "Model", kind: .string),
            MereRunCapabilityOption(flag: "--final", label: "Final", kind: .boolean)
        ],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: ["--model"],
            defaultModels: [.init(whenAny: [.init(flag: "--final")], models: ["reel-any"]), .always("reel-draft")],
            families: [
                .init(ReelFamily.draft, title: "Draft", models: ["reel-draft"]),
                .init(ReelFamily.full, title: "Full", models: [])
            ],
            identifiedModels: ["reel-any"]
        )
    )
    let installed: (String) -> MereRunModelIdentification? = { model in
        switch model {
        case "reel-any": .family("full")
        case "Reel-Any": .managedModel("reel-any")
        default: nil
        }
    }
    let resolve = { (arguments: [String], identify: (String) -> MereRunModelIdentification?) in
        reel.resolveFamily(MereRunCommandInvocation(capability: reel, arguments: arguments), identify: identify)
    }
    #expect(resolve(["--model", "reel-any"], installed) == .family(id: "full", model: "reel-any", source: .identified))
    #expect(resolve(["--model", "Reel-Any"], installed) == .family(id: "full", model: "reel-any", source: .identified))
    #expect(resolve(["--final"], installed) == .family(id: "full", model: "reel-any", source: .defaultModel))
    #expect(resolve(["--final"], { _ in nil }) == .unidentified(model: "reel-any"))
    #expect(resolve(["--model", "reel-any"], { _ in nil }) == .unidentified(model: "reel-any"))
    #expect(resolve([], { _ in nil }) == .family(id: "draft", model: "reel-draft", source: .defaultModel))

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let json = String(decoding: try encoder.encode(try #require(reel.routing)), as: UTF8.self)
    #expect(json.contains(#""identified_models":["reel-any"]"#))
    #expect(try JSONDecoder().decode(MereRunCapabilityRouting.self, from: Data(json.utf8)) == reel.routing)
    #expect(!String(decoding: try encoder.encode(try #require(clip.routing)), as: UTF8.self).contains("identified_models"))
}

/// One model in two families split by a flag's absence: the embedded-adapter family without the
/// flag, the adapter family with it.
@Test func anAbsentSelectorSplitsOneModelBetweenTwoFamilies() throws {
    enum TurboFamily: String, MereRunFamilyID { case embedded, adapter }
    let turbo = MereRunCommandCapability(
        id: "turbo.render", command: ["turbo", "render"], title: "Turbo", summary: "A test capability.",
        options: [
            MereRunCapabilityOption(flag: "--model", label: "Model", kind: .string),
            MereRunCapabilityOption(flag: "--adapter", label: "Adapter", kind: .string),
            MereRunCapabilityOption(flag: "--steps", label: "Steps", kind: .integer)
                .scoped(TurboFamily.rule(.embedded, values: ["5"]))
        ],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: ["--model"],
            defaultModels: [.always("turbo-fast")],
            families: [
                .init(TurboFamily.embedded, title: "Embedded", models: ["turbo-fast"], selectors: [.absent("--adapter")]),
                .init(TurboFamily.adapter, title: "Adapter", models: ["turbo-fast"], selectors: [.init(flag: "--adapter")])
            ]
        )
    )
    let report = { (arguments: [String]) in
        turbo.resolutionReport(MereRunCommandInvocation(capability: turbo, arguments: arguments))
    }
    #expect(report([]).family == "embedded")
    #expect(report(["--model", "turbo-fast"]).family == "embedded")
    #expect(report(["--model", "turbo-fast", "--steps", "9"]).violations
        == ["--steps 9 is not supported by Embedded; it runs 5. Remove --steps or pass 5."])
    let lifted = report(["--model", "turbo-fast", "--adapter", "a.safetensors", "--steps", "9"])
    #expect(lifted.family == "adapter" && lifted.source == .selector && lifted.violations.isEmpty)

    let conditions = try #require(turbo.routing).families.map { try #require($0.selectors.first) }
    #expect(conditions[0].excludes(conditions[1]) && conditions[1].excludes(conditions[0]))
    #expect(!conditions[0].excludes(.init(flag: "--adapter", values: ["a"])), "an omitted value can read as its default")
    #expect(!conditions[0].excludes(.absent("--adapter")))

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let json = String(decoding: try encoder.encode(conditions), as: UTF8.self)
    #expect(json == #"[{"absent":true,"flag":"--adapter"},{"flag":"--adapter"}]"#)
    #expect(try JSONDecoder().decode([MereRunFlagCondition].self, from: Data(json.utf8)) == conditions)
}

/// When the selectors outrank the model, a listed model whose family's selectors fail runs as if
/// no model were named, and the report warns about it however the model is spelled.
@Test func selectorsThatOverrideTheModelReplaceItWithAWarning() throws {
    enum ListenFamily: String, MereRunFamilyID { case fast, deep }
    let listener = MereRunCommandCapability(
        id: "ear.listen", command: ["ear", "listen"], title: "Listen", summary: "A test capability.",
        options: [
            MereRunCapabilityOption(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            MereRunCapabilityOption(flag: "--engine", label: "Engine", kind: .choice,
                                    choices: ["auto", "fast", "deep"], defaultValue: "auto")
                .scoped(ListenFamily.rule(.deep, values: ["auto", "deep"], severity: .warning)),
            MereRunCapabilityOption(flag: "--task", label: "Task", kind: .choice,
                                    choices: ["hear", "explain"], defaultValue: "hear")
        ],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: ["--model"],
            defaultModels: [
                .init(whenAny: [.init(flag: "--task", values: ["explain"]), .init(flag: "--engine", values: ["deep"])],
                      models: ["ear-deep"]),
                .always("ear-fast")
            ],
            families: [
                .init(ListenFamily.fast, title: "Fast", models: ["ear-fast"],
                      selectors: [.init(flag: "--task", values: ["hear"]), .init(flag: "--engine", values: ["auto", "fast"])]),
                .init(ListenFamily.deep, title: "Deep", models: ["ear-deep"],
                      selectors: [.init(flag: "--engine", values: ["auto", "deep"])])
            ],
            selectorsOverrideModel: true
        )
    )
    let identify: (String) -> MereRunModelIdentification? = { $0 == "EAR-FAST" ? .managedModel("ear-fast") : nil }
    let report = { (arguments: [String]) in
        listener.resolutionReport(MereRunCommandInvocation(capability: listener, arguments: arguments), identify: identify)
    }
    let honored = report(["--model", "ear-deep"])
    #expect(honored.family == "deep" && honored.source == .model && honored.warnings.isEmpty)

    let replaced = report(["--model", "ear-fast", "--task", "explain"])
    #expect(replaced.family == "deep" && replaced.model == "ear-deep" && replaced.source == .defaultModel)
    #expect(replaced.violations.isEmpty)
    #expect(replaced.warnings == ["--model ear-fast has no effect: the other options select Deep."])
    #expect(report(["-m", "EAR-FAST", "--task", "explain"]).warnings
        == ["--model EAR-FAST has no effect: the other options select Deep."])

    let named = report(["--model", "ear-deep", "--engine", "fast"])
    #expect(named.family == "fast" && named.model == "ear-fast")
    #expect(named.warnings == ["--model ear-deep has no effect: the other options select Fast."])

    // The default rules decide without rechecking the chosen family's selectors: the task
    // outranks an explicit engine, which then draws the rule's warning.
    let outranked = report(["--engine", "fast", "--task", "explain"])
    #expect(outranked.family == "deep" && outranked.violations.isEmpty)
    #expect(outranked.warnings == ["--engine fast has no effect with Deep; use auto or deep."])
    #expect(report(["--model", "ear-deep", "--engine", "fast", "--task", "explain"]).warnings
        == ["--engine fast has no effect with Deep; use auto or deep."])

    // The command's own router outranks the declared rules; the model follows the routed family.
    let routed = { (arguments: [String], family: String?) in
        listener.resolutionReport(
            MereRunCommandInvocation(capability: listener, arguments: arguments), identify: identify, routedFamily: { family }
        )
    }
    let rerouted = routed(["--model", "ear-fast", "--engine", "fast"], "deep")
    #expect(rerouted.family == "deep" && rerouted.model == "ear-deep" && rerouted.source == .defaultModel)
    #expect(rerouted.warnings == [
        "--model ear-fast has no effect: the other options select Deep.",
        "--engine fast has no effect with Deep; use auto or deep."
    ])
    #expect(routed(["--model", "ear-fast"], "fast") == report(["--model", "ear-fast"]))
    #expect(routed(["--model", "ear-fast"], "unknown") == report(["--model", "ear-fast"]))
    let local = routed(["--model", "/ears/deep"], "deep")
    #expect(local.family == "deep" && local.model == "/ears/deep" && local.source == .identified)
    #expect(routed(["-m", "EAR-FAST"], "deep").warnings == ["--model EAR-FAST has no effect: the other options select Deep."])

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let routing = try encoder.encode(try #require(listener.routing))
    #expect(String(decoding: routing, as: UTF8.self).contains(#""selectors_override_model":true"#))
    #expect(try JSONDecoder().decode(MereRunCapabilityRouting.self, from: routing) == listener.routing)
}

@Test func violationsCoverEveryScopeKindWithOneMessage() {
    let messages = { (family: String, arguments: [String]) in
        clip.violations(MereRunCommandInvocation(capability: clip, arguments: arguments), family: family)
            .map { "\($0.severity.rawValue): \($0.message)" }
    }
    #expect(messages("remote", ["--steps", "4"])
        == ["error: --steps is not supported by Remote. It applies to Quick, Full and Wide."])
    #expect(messages("quick", ["--cfg", "3"]) == ["warning: --cfg has no effect with Quick. It applies to Full."])
    #expect(messages("wide", ["--cfg", "3", "--image", "a"]) == ["error: --cfg is not supported by Wide. It applies to Full."])
    #expect(messages("quick", ["-s", "8"])
        == ["error: --steps 8 is not supported by Quick; it runs 4. Remove --steps or pass 4."])
    #expect(messages("quick", ["--steps", "4"]).isEmpty)
    #expect(messages("quick", ["--steps", "4.0"]).isEmpty, "a numeric value matches by number, as the CLI parses it")
    #expect(messages("quick", ["--steps", "04"]).isEmpty, "numbers match by value")
    #expect(messages("full", ["--steps", "70"])
        == ["error: --steps 70 is not supported by Full; use a value from 10 to 60."])
    #expect(messages("wide", []) == ["error: Wide requires --image."])
    #expect(messages("full", ["--image", "a", "--image", "b", "--image", "c"])
        == ["warning: Full takes --image at most 2 times; got 3."])
    #expect(messages("full", ["--mode", "fast"])
        == ["error: --mode fast is not supported by Full; use slow or exact."])
    #expect(messages("quick", ["--mode", "fast", "--steps=4", "--hq"]).isEmpty)
    #expect(messages("quick", ["--strength", "0.50"])
        == ["warning: --strength has no effect with Quick. It applies to Full."])
    #expect(messages("quick", ["--strength", "0.7"])
        == ["error: --strength is not supported by Quick. It applies to Full."])
    #expect(messages("wide", ["--strength", "0.5", "--image", "a"])
        == ["error: --strength is not supported by Wide. It applies to Full."])

    let report = clip.resolutionReport(invocation("--model", "clip-quick", "--cfg", "2", "--steps", "9"))
    #expect(report.family == "quick" && report.familyTitle == "Quick" && report.source == .model)
    #expect(report.violations == ["--steps 9 is not supported by Quick; it runs 4. Remove --steps or pass 4."])
    #expect(report.warnings == ["--cfg has no effect with Quick. It applies to Full."])
}

/// A rule on a family that ignores an option lists the values it lets through with a warning,
/// and choice spellings the CLI accepts resolve like the choice in rules and default conditions.
@Test func ignoredValueRulesAndChoiceSpellingsFollowTheCLI() {
    enum PressFamily: String, MereRunFamilyID { case fine, rough }
    let press = MereRunCommandCapability(
        id: "press.run", command: ["press", "run"], title: "Press", summary: "A test capability.",
        options: [
            MereRunCapabilityOption(flag: "--model", label: "Model", kind: .string),
            MereRunCapabilityOption(
                flag: "--recipe", label: "Recipe", kind: .choice, choices: ["fine-a", "rough-a"],
                choiceSpellings: .init(ignoresCase: true, aliases: ["old-fine": "fine-a"])
            ).scoped(PressFamily.rule(.rough, values: ["rough-a"])),
            MereRunCapabilityOption(flag: "--passes", label: "Passes", kind: .integer)
                .scoped(PressFamily.only(.fine, ignoredBy: [.rough]), .rule(.rough, range: .init(min: 8, max: 8))),
            MereRunCapabilityOption(flag: "--grain", label: "Grain", kind: .number)
                .scoped(PressFamily.only(.fine, ignoredBy: [.rough]))
        ],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: ["--model"],
            defaultModels: [
                .init(whenAny: [.init(flag: "--recipe", values: ["fine-a"])], models: ["press-fine"]),
                .always("press-rough")
            ],
            families: [
                .init(PressFamily.fine, title: "Fine", models: ["press-fine"]),
                .init(PressFamily.rough, title: "Rough", models: ["press-rough"])
            ]
        )
    )
    let report = { (arguments: [String]) in
        press.resolutionReport(MereRunCommandInvocation(capability: press, arguments: arguments))
    }
    for spelling in ["fine-a", "FINE-A", " Old-Fine "] {
        #expect(report(["--recipe", spelling]).family == "fine", "\(spelling)")
    }
    #expect(report(["--recipe", "Rough-A"]).family == "rough")
    #expect(report(["--recipe", "OLD-FINE", "--model", "press-rough"]).violations
        == ["--recipe OLD-FINE is not supported by Rough; it runs rough-a. Remove --recipe or pass rough-a."])
    #expect(report(["--recipe", "ROUGH-A", "--model", "press-rough"]).violations.isEmpty)

    let tolerated = report(["--passes", "8"])
    #expect(tolerated.violations.isEmpty && tolerated.warnings == ["--passes has no effect with Rough. It applies to Fine."])
    #expect(report(["--passes", "3"]).violations == ["--passes is not supported by Rough. It applies to Fine."])
    #expect(report(["--grain", "3"]).warnings == ["--grain has no effect with Rough. It applies to Fine."])
    #expect(!press.options(forFamily: "rough").map(\.flag).contains("--passes"), "still hidden on the ignoring family")
    let fine = report(["--model", "press-fine", "--passes", "3"])
    #expect(fine.violations.isEmpty && fine.warnings.isEmpty)
}

@Test func aBooleanSelectorHoldsOnItsPresence() {
    enum ReadFamily: String, MereRunFamilyID { case single, compare }
    let reader = MereRunCommandCapability(
        id: "reader.read", command: ["reader", "read"], title: "Read", summary: "A test capability.",
        options: [
            MereRunCapabilityOption(flag: "--compare", label: "Compare", kind: .boolean),
            MereRunCapabilityOption(flag: "--model", label: "Model", kind: .string)
        ],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: [],
            families: [
                .init(ReadFamily.single, title: "Single", models: [], selectors: [.init(flag: "--compare", values: ["false"])]),
                .init(ReadFamily.compare, title: "Compare", models: [], selectors: [.init(flag: "--compare", values: ["true"])])
            ]
        )
    )
    let read = { (arguments: [String]) in
        reader.resolveFamily(MereRunCommandInvocation(capability: reader, arguments: arguments))
    }
    #expect(read([]) == .family(id: "single", model: nil, source: .selector))
    #expect(read(["--compare"]) == .family(id: "compare", model: nil, source: .selector))
    let selectors = reader.routing?.families.flatMap(\.selectors) ?? []
    #expect(selectors.map(reader.arguments(satisfying:)) == [[], ["--compare"]])
    #expect(reader.arguments(satisfying: .init(flag: "--model", values: ["a", "b"])) == ["--model", "a"])
    #expect(reader.arguments(satisfying: .init(flag: "--model")) == ["--model", "value"])
    #expect(reader.arguments(satisfying: .absent("--model")).isEmpty)
}

@Test func anEmptyTextValueOnlyWarns() {
    enum NoteFamily: String, MereRunFamilyID { case plain, noted }
    let note = MereRunCommandCapability(
        id: "note.write", command: ["note", "write"], title: "Write", summary: "A test capability.",
        options: [
            MereRunCapabilityOption(flag: "--model", label: "Model", kind: .string),
            MereRunCapabilityOption(flag: "--note", label: "Note", kind: .string).scoped(NoteFamily.only(.noted)),
            MereRunCapabilityOption(flag: "--attachment", label: "Attachment", kind: .file).scoped(NoteFamily.only(.noted))
        ],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: ["--model"],
            families: [
                .init(NoteFamily.plain, title: "Plain", models: ["note-plain"]),
                .init(NoteFamily.noted, title: "Noted", models: ["note-noted"])
            ]
        )
    )
    let messages = { (arguments: [String]) in
        note.violations(MereRunCommandInvocation(capability: note, arguments: arguments), family: "plain")
            .map { "\($0.severity.rawValue): \($0.message)" }
    }
    #expect(messages(["--note", ""]) == ["warning: --note has no effect with Plain. It applies to Noted."])
    #expect(messages(["--note", "x"]) == ["error: --note is not supported by Plain. It applies to Noted."])
    #expect(messages(["--attachment", ""]) == ["error: --attachment is not supported by Plain. It applies to Noted."],
            "an empty file still counts as passed")
}

@Test func optionsForAFamilyApplyItsRules() throws {
    #expect(clip.options(forFamily: nil) == clip.options)
    let remote = clip.options(forFamily: "remote").map(\.flag)
    #expect(remote == ["--model", "--model-root", "--backend", "--hq", "--mode"])
    let quick = clip.options(forFamily: "quick")
    #expect(!quick.map(\.flag).contains("--cfg"), "an ignored option is not part of the family's surface")
    let quickSteps = try #require(quick.first { $0.flag == "--steps" })
    #expect(quickSteps.defaultValue == "4" && quickSteps.familyRules.map(\.values) == [["4"]])
    let full = clip.options(forFamily: "full")
    let mode = try #require(full.first { $0.flag == "--mode" })
    #expect(mode.choices == ["slow", "exact"] && mode.defaultValue == "slow")
    #expect(try #require(full.first { $0.flag == "--steps" }).range == .init(min: 10, max: 60))
    #expect(try #require(clip.options(forFamily: "wide").first { $0.flag == "--image" }).required)
}

@Test func typedScopesSerializeInDeclarationOrder() throws {
    let steps = try #require(clipOptions.first { $0.flag == "--steps" })
    #expect(steps.families == ["quick", "full", "wide"])
    #expect(steps.familyRules.map(\.family) == ["quick", "full"])
    let image = try #require(clipOptions.first { $0.flag == "--image" })
    #expect(image.families == ["full", "wide"])

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let json = String(decoding: try encoder.encode(image), as: UTF8.self)
    #expect(json.contains(#""families":["full","wide"]"#))
    #expect(json.contains(#""family_rules":[{"family":"wide","required":true},{"family":"full","max_count":2,"severity":"warning"}]"#))
    #expect(try JSONDecoder().decode(MereRunCapabilityOption.self, from: Data(json.utf8)) == image)
    let cfg = String(decoding: try encoder.encode(try #require(clipOptions.first { $0.flag == "--cfg" })), as: UTF8.self)
    #expect(cfg.contains(#""ignored_by":["quick"]"#))
}

@Test func routingSerializesAdditively() throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let unrouted = String(decoding: try encoder.encode(plain), as: UTF8.self)
    #expect(!unrouted.contains("routing") && !unrouted.contains("family_rules") && !unrouted.contains("ignored_by"))

    let routing = String(decoding: try encoder.encode(try #require(clip.routing)), as: UTF8.self)
    #expect(routing.contains(#""model_flags":["--model-root","--model"]"#))
    #expect(routing.contains(#""default_models":[{"models":["clip-full"],"when_any":[{"flag":"--hq"}]},{"models":["clip-quick"]}]"#))
    #expect(routing.contains(#""excluded_models":[{"id":"clip-lm""#))
    #expect(routing.contains(#"{"id":"full","models":["clip-full"],"title":"Full"}"#))
    #expect(!routing.contains("selectors_override_model"))
    #expect(try JSONDecoder().decode(MereRunCapabilityRouting.self, from: Data(routing.utf8)) == clip.routing)

    let document = try encoder.encode(MereRunCapabilityCatalog.document)
    #expect(try JSONDecoder().decode(MereRunCapabilityDocument.self, from: document) == MereRunCapabilityCatalog.document)

    let report = MereRunFamilyResolutionReport(
        capability: "clip.render", family: "quick", familyTitle: "Quick", model: nil, source: .defaultModel,
        violations: [], warnings: ["w"]
    )
    let reportJSON = String(decoding: try encoder.encode(report), as: UTF8.self)
    #expect(reportJSON == #"{"capability":"clip.render","family":"quick","family_title":"Quick","source":"default","violations":[],"warnings":["w"]}"#)
}

// MARK: - Reading argv

@Test func invocationReadsArgvTheWayArgumentParserDoes() {
    let read = MereRunCommandInvocation(
        capability: clip,
        arguments: ["in.png", "-m", "clip-full", "--steps=12", "--hq", "--cfg", "-1.5", "--image", "a", "--image=b",
                    "--unknown", "stray", "--", "--steps", "9"]
    )
    #expect(read.values == [
        "--model": ["clip-full"], "--steps": ["12"], "--hq": [], "--cfg": ["-1.5"], "--image": ["a", "b"]
    ])
    #expect(read.positionals == ["in.png", "stray", "--steps", "9"])
    #expect(read.undeclared == ["--unknown"])
    #expect(read.contains("--hq") && !read.contains("--mode"))
    #expect(read.value("--image") == "b")
}

@Test func commandLinesMatchTheLongestCatalogedPath() throws {
    let face = try #require(MereRunCapabilityCatalog.capability(forCommandLine: ["vision", "face", "detect", "a.png", "-m", "x"]))
    #expect(face.capability.id == "vision.face.detect")
    #expect(face.arguments == ["a.png", "-m", "x"])
    #expect(MereRunCapabilityCatalog.capability(forCommandLine: ["vision"]) == nil)
    #expect(MereRunCapabilityCatalog.capability(forCommandLine: ["nope", "run"]) == nil)
}
