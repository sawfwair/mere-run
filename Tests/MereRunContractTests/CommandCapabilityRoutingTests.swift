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
    }
}

@Test func noModelBelongsToTwoFamiliesWhoseSelectorsCanBothMatch() {
    for (capability, routing) in routed {
        for (index, first) in routing.families.enumerated() {
            for second in routing.families.dropFirst(index + 1) {
                let shared = Set(first.models).intersection(second.models)
                guard !shared.isEmpty || routing.routesBySelectors else { continue }
                let disjoint = first.selectors.contains { left in
                    second.selectors.contains { right in
                        left.flag == right.flag && Set(left.values ?? []).isDisjoint(with: right.values ?? [])
                            && left.values != nil && right.values != nil
                    }
                }
                #expect(disjoint, "\(capability.id): \(first.id) and \(second.id) can both match \(shared.sorted())")
            }
        }
    }
}

@Test func everyFamilyIsReachable() {
    for (capability, routing) in routed {
        for family in routing.families where family.models.isEmpty && !routing.routesBySelectors {
            #expect(
                routing.defaultModels.contains { $0.family == family.id },
                "\(capability.id) \(family.id) lists no model, so a default rule has to name it"
            )
        }
    }
}

@Test func everyDefaultRuleResolvesToOneFamilyOnMacOS() {
    for (capability, routing) in routed {
        for rule in routing.defaultModels where rule.applies(on: "macos") {
            let families = rule.family.map { [$0] }
                ?? Array(Set(rule.models.flatMap { model in routing.families.filter { $0.models.contains(model) }.map(\.id) }))
            #expect(families.count == 1, "\(capability.id): default \(rule.models) resolves to \(families)")
            #expect(families.allSatisfy { routing.family(id: $0) != nil }, "\(capability.id): default names an unknown family")
        }
        let blank = MereRunCommandInvocation(capability: capability, arguments: [])
        guard case .family = capability.resolveFamily(blank) else {
            Issue.record("\(capability.id): a blank command line must resolve to a family on macOS")
            continue
        }
    }
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
                #expect(used.contains(rule.family), "\(ruleContext): the family must use the option")
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
    case .boolean: false
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
        .scoped(ClipFamily.rule(.full, values: ["slow", "exact"], defaultValue: "slow"))
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
    #expect(MereRunCapabilityCatalog.textChat.resolveFamily(invocation()) == .unrouted)
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
    #expect(messages("full", ["--steps", "70"])
        == ["error: --steps 70 is not supported by Full; use a value from 10 to 60."])
    #expect(messages("wide", []) == ["error: Wide requires --image."])
    #expect(messages("full", ["--image", "a", "--image", "b", "--image", "c"])
        == ["warning: Full takes --image at most 2 times; got 3."])
    #expect(messages("full", ["--mode", "fast"])
        == ["error: --mode fast is not supported by Full; use slow or exact."])
    #expect(messages("quick", ["--mode", "fast", "--steps=4", "--hq"]).isEmpty)

    let report = clip.resolutionReport(invocation("--model", "clip-quick", "--cfg", "2", "--steps", "9"))
    #expect(report.family == "quick" && report.familyTitle == "Quick" && report.source == .model)
    #expect(report.violations == ["--steps 9 is not supported by Quick; it runs 4. Remove --steps or pass 4."])
    #expect(report.warnings == ["--cfg has no effect with Quick. It applies to Full."])
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
    let unrouted = String(decoding: try encoder.encode(MereRunCapabilityCatalog.textChat), as: UTF8.self)
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
