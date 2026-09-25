import Foundation

/// Which runtime family runs an invocation.
public enum MereRunFamilyResolution: Equatable, Sendable {
    /// The capability has no routing.
    case unrouted
    case family(id: String, model: String?, source: Source)
    /// A model the contract does not list (a folder, an alias, an unknown id), or a default the
    /// machine chooses between families, that the identifier could not answer. Callers show the
    /// full surface; the command decides when it runs.
    case unidentified(model: String)
    case excluded(MereRunExcludedModel)
    /// A listed model whose selectors match no family (`--backend qwen` with a Parakeet id).
    case unmatched(model: String?, detail: String)

    public enum Source: String, Codable, Sendable {
        case model
        case defaultModel = "default"
        case selector
        case identified
    }
}

/// What the CLI knows about a model the contract does not list, from inspecting it.
public enum MereRunModelIdentification: Equatable, Sendable {
    /// An alias or upstream repository id of this managed model; it resolves like the id itself.
    case managedModel(String)
    /// A local model of this family of the capability.
    case family(String)
}

extension MereRunFamilyResolutionReport {
    /// The resolution this report describes: what a shell that asked `catalog resolve` about a
    /// command line takes as its family. `capability` is the one the report is about.
    public func resolution(in capability: MereRunCommandCapability) -> MereRunFamilyResolution {
        switch source {
        case .unrouted:
            return .unrouted
        case .unidentified:
            return .unidentified(model: model ?? "")
        case .excluded:
            guard let model, let excluded = capability.routing?.excludedModel(id: model) else {
                return .unidentified(model: model ?? "")
            }
            return .excluded(excluded)
        case .unmatched:
            return .unmatched(model: model, detail: violations.first ?? "")
        case .model, .defaultModel, .selector, .identified:
            guard let family else { return .unidentified(model: model ?? "") }
            let resolved: MereRunFamilyResolution.Source = switch source {
            case .defaultModel: .defaultModel
            case .selector: .selector
            case .identified: .identified
            default: .model
            }
            return .family(id: family, model: model, source: resolved)
        }
    }
}

/// One way an invocation leaves its family's scope.
public struct MereRunOptionViolation: Equatable, Sendable {
    public enum Severity: String, Codable, Sendable {
        /// The CLI refuses to run.
        case error
        /// The CLI runs and the option has no effect.
        case warning
    }

    public enum Kind: Equatable, Sendable {
        /// The family does not use the option; `supportedBy` names the families that do.
        case unsupported(supportedBy: [String])
        case valueNotAllowed(allowed: [String])
        case outOfRange(MereRunCapabilityRange)
        case missingRequired
        case tooMany(max: Int)
    }

    public let flag: String
    public let kind: Kind
    public let severity: Severity
    /// The sentence the CLI prints and shells show.
    public let message: String

    public init(flag: String, kind: Kind, severity: Severity, message: String) {
        self.flag = flag
        self.kind = kind
        self.severity = severity
        self.message = message
    }
}

/// Everything the CLI's capability gate decides for one invocation; `catalog resolve --json`
/// prints it.
public struct MereRunFamilyResolutionReport: Codable, Equatable, Sendable {
    public enum Source: String, Codable, Sendable {
        case model
        case defaultModel = "default"
        case selector
        case identified
        case unidentified
        case excluded
        case unmatched
        case unrouted
    }

    public let capability: String
    public let family: String?
    public let familyTitle: String?
    public let model: String?
    public let source: Source
    /// Why the gate refuses the run. Empty means it runs.
    public let violations: [String]
    /// Options the gate reports as having no effect for the family; the run continues.
    public let warnings: [String]

    enum CodingKeys: String, CodingKey {
        case capability, family, model, source, violations, warnings
        case familyTitle = "family_title"
    }

    public init(
        capability: String,
        family: String?,
        familyTitle: String?,
        model: String?,
        source: Source,
        violations: [String],
        warnings: [String]
    ) {
        self.capability = capability
        self.family = family
        self.familyTitle = familyTitle
        self.model = model
        self.source = source
        self.violations = violations
        self.warnings = warnings
    }
}

extension MereRunCommandCapability {
    /// The one family resolver. `identify` answers models the contract does not list.
    /// `chooseDefault` picks among a default rule's candidates when they span families and the
    /// CLI chooses by machine, returning the candidate this machine runs. `routedFamily` is the
    /// family the command's own router picks for the whole command line, where the declared rules
    /// only approximate it (speech transcribe routes an unrecognized `--language` to Qwen3-ASR).
    /// Its answer wins over the rules; `nil` keeps them.
    public func resolveFamily(
        _ invocation: MereRunCommandInvocation,
        platform: String = "macos",
        identify: (String) -> MereRunModelIdentification? = { _ in nil },
        chooseDefault: ([String]) -> String? = { _ in nil },
        routedFamily: () -> String? = { nil }
    ) -> MereRunFamilyResolution {
        guard let routing else { return .unrouted }
        let declared = declaredFamily(
            invocation, routing: routing, platform: platform, identify: identify, chooseDefault: chooseDefault
        )
        // A router picks between runs; a model the command refuses stays refused.
        switch declared {
        case .excluded, .unmatched: return declared
        case .unrouted, .family, .unidentified: break
        }
        guard let routed = routedFamily().flatMap(routing.family(id:)) else { return declared }
        if case .family(let id, _, _) = declared, id == routed.id { return declared }
        return routerChoice(routed, invocation, routing: routing, platform: platform, identify: identify)
    }

    private func declaredFamily(
        _ invocation: MereRunCommandInvocation,
        routing: MereRunCapabilityRouting,
        platform: String,
        identify: (String) -> MereRunModelIdentification?,
        chooseDefault: ([String]) -> String?
    ) -> MereRunFamilyResolution {
        if routing.routesBySelectors {
            return resolveBySelectors(invocation, routing: routing, platform: platform, identify: identify)
        }
        guard let model = modelValue(invocation, flags: routing.modelFlags) else {
            return resolveDefault(
                invocation, routing: routing, platform: platform, identify: identify, chooseDefault: chooseDefault
            )
        }
        return resolve(
            model: model, invocation, routing: routing, platform: platform, identify: identify,
            chooseDefault: chooseDefault, allowIdentify: true
        )
    }

    /// The options `family` uses, each with its rule applied: narrowed `choices`, the family's
    /// default and `range`, and `required`. A rule's allowed values stay readable on the
    /// returned option's `familyRules`. `nil` family: every option unchanged.
    public func options(forFamily family: String?) -> [MereRunCapabilityOption] {
        guard let family else { return options }
        return options.filter { $0.families?.contains(family) ?? true }.map { option in
            let rule = option.familyRules.first { $0.family == family }
            let fixed = rule?.values?.count == 1 ? rule?.values?.first : nil
            return MereRunCapabilityOption(
                flag: option.flag, aliases: option.aliases, label: option.label, kind: option.kind,
                required: option.required || rule?.required == true, repeatable: option.repeatable,
                choices: option.kind == .choice ? rule?.values ?? option.choices : option.choices,
                defaultValue: rule?.defaultValue ?? fixed ?? option.defaultValue, group: option.group,
                tier: option.tier, range: rule?.range ?? option.range, dependsOn: option.dependsOn,
                familyRules: rule.map { [$0] } ?? [], choiceSpellings: option.choiceSpellings
            )
        }
    }

    /// Every way `invocation` leaves `family`'s scope, errors and warnings, in option order.
    /// `identify` names the managed model an alias stands for, so a named model the selectors
    /// override is reported however it is spelled.
    public func violations(
        _ invocation: MereRunCommandInvocation,
        family: String,
        identify: (String) -> MereRunModelIdentification? = { _ in nil }
    ) -> [MereRunOptionViolation] {
        guard let routing, let runtime = routing.family(id: family) else { return [] }
        return options.flatMap { option -> [MereRunOptionViolation] in
            let rule = option.familyRules.first { $0.family == family }
            guard let values = invocation.values[option.flag] else {
                guard rule?.required == true else { return [] }
                return [MereRunOptionViolation(
                    flag: option.flag, kind: .missingRequired, severity: .error,
                    message: "\(runtime.title) requires \(option.flag)."
                )]
            }
            if routing.selectorsOverrideModel, routing.modelFlags.contains(option.flag) {
                return overriddenModel(option.flag, invocation, routing: routing, family: runtime, identify: identify)
            }
            if let families = option.families, !families.contains(family) {
                let titles = families.compactMap { routing.family(id: $0)?.title }
                // A family's rule on an option it ignores lists the values it tolerates, the way a
                // value-based check in the CLI lets only the value it runs with through; any other
                // value is refused with the same message as an option the family rejects.
                let refused = rule.map { !ruleViolations(option, values: values, rule: $0, family: runtime).isEmpty } ?? false
                let ignored = option.ignoredBy.contains(family) && !refused || option.readsAsOmitted(values)
                return [unsupported(option.flag, family: runtime, ignored: ignored, usedBy: titles)]
            }
            guard let rule else { return [] }
            return ruleViolations(option, values: values, rule: rule, family: runtime)
        }
    }

    /// The gate's whole decision for `invocation`.
    public func resolutionReport(
        _ invocation: MereRunCommandInvocation,
        platform: String = "macos",
        identify: (String) -> MereRunModelIdentification? = { _ in nil },
        chooseDefault: ([String]) -> String? = { _ in nil },
        routedFamily: () -> String? = { nil }
    ) -> MereRunFamilyResolutionReport {
        report(
            for: resolveFamily(
                invocation, platform: platform, identify: identify, chooseDefault: chooseDefault, routedFamily: routedFamily
            ),
            invocation,
            identify: identify
        )
    }

    /// The gate's report for `invocation` once its family is resolved: what a shell that asked
    /// `catalog resolve` for the family, and scopes the rest of the command line itself, shows.
    public func report(
        for resolution: MereRunFamilyResolution,
        _ invocation: MereRunCommandInvocation,
        identify: (String) -> MereRunModelIdentification? = { _ in nil }
    ) -> MereRunFamilyResolutionReport {
        let report = { (family: MereRunRuntimeFamily?, model: String?, source: MereRunFamilyResolutionReport.Source,
                        violations: [String], warnings: [String]) in
            MereRunFamilyResolutionReport(
                capability: id, family: family?.id, familyTitle: family?.title, model: model, source: source,
                violations: violations, warnings: warnings
            )
        }
        switch resolution {
        case .unrouted:
            return report(nil, nil, .unrouted, [], [])
        case .unidentified(let model):
            return report(nil, model, .unidentified, [], [])
        case .excluded(let excluded):
            let message = "\(excluded.id) can't run \(command.joined(separator: " ")): \(excluded.reason)"
            return report(nil, excluded.id, .excluded, [message], [])
        case let .unmatched(model, detail):
            return report(nil, model, .unmatched, [detail], [])
        case let .family(familyID, model, source):
            let found = violations(invocation, family: familyID, identify: identify)
            let source: MereRunFamilyResolutionReport.Source = switch source {
            case .model: .model
            case .defaultModel: .defaultModel
            case .selector: .selector
            case .identified: .identified
            }
            return report(
                routing?.family(id: familyID), model, source,
                found.filter { $0.severity == .error }.map(\.message),
                found.filter { $0.severity == .warning }.map(\.message)
            )
        }
    }
}

// MARK: - Resolution

extension MereRunCommandCapability {
    /// The shortest arguments that make `condition` hold: the flag and its first allowed value;
    /// for a presence condition, the flag alone on a Boolean and the flag with the option's
    /// default, first choice, or a placeholder otherwise; and nothing for a Boolean that must be
    /// off or a flag that must be absent. Generators and coverage tests build argv from it.
    public func arguments(satisfying condition: MereRunFlagCondition) -> [String] {
        if condition.absent { return [] }
        let option = options.first { $0.flag == condition.flag }
        guard option?.kind == .boolean else {
            let value = condition.values?.first ?? option?.defaultValue ?? option?.choices.first ?? "value"
            return [condition.flag, value]
        }
        return condition.values?.first == "false" ? [] : [condition.flag]
    }

    private func modelValue(_ invocation: MereRunCommandInvocation, flags: [String]) -> String? {
        flags.lazy.compactMap { invocation.value($0) }.first { !$0.isEmpty }
    }

    /// The model `family` runs when the command's router picked it over the declared rules: the
    /// named model when the family lists it or no family does (a local path, an unlisted id), and
    /// otherwise the family's default.
    private func routerChoice(
        _ family: MereRunRuntimeFamily,
        _ invocation: MereRunCommandInvocation,
        routing: MereRunCapabilityRouting,
        platform: String,
        identify: (String) -> MereRunModelIdentification?
    ) -> MereRunFamilyResolution {
        if let named = modelValue(invocation, flags: family.modelFlag.map { [$0] } ?? routing.modelFlags) {
            var canonical = named
            if case .managedModel(let managed)? = identify(named) { canonical = managed }
            if family.models.contains(canonical) {
                return .family(id: family.id, model: canonical, source: .model)
            }
            if !routing.families.contains(where: { $0.models.contains(canonical) }) {
                return .family(id: family.id, model: named, source: .identified)
            }
        }
        let rule = routing.defaultModels.first { rule in
            rule.applies(on: platform)
                && (rule.family == family.id || (!rule.models.isEmpty && rule.models.allSatisfy(family.models.contains)))
        }
        return .family(id: family.id, model: rule?.models.first, source: .defaultModel)
    }

    private func resolve(
        model: String,
        _ invocation: MereRunCommandInvocation,
        routing: MereRunCapabilityRouting,
        platform: String,
        identify: (String) -> MereRunModelIdentification?,
        chooseDefault: ([String]) -> String?,
        allowIdentify: Bool
    ) -> MereRunFamilyResolution {
        if let excluded = routing.excludedModel(id: model) {
            return .excluded(excluded)
        }
        // What is installed decides first for a model whose family depends on it.
        if routing.identifiedModels.contains(model), case .family(let id)? = identify(model), let family = routing.family(id: id) {
            guard selectorsHold(family, invocation) else {
                return .unmatched(model: model, detail: unmatchedDetail(model: model, candidates: [family]))
            }
            return .family(id: id, model: model, source: .identified)
        }
        let candidates = routing.families.filter { $0.models.contains(model) }
        if !candidates.isEmpty {
            let matching = candidates.filter { selectorsHold($0, invocation) }
            guard let family = matching.first else {
                if routing.selectorsOverrideModel {
                    return resolveDefault(
                        invocation, routing: routing, platform: platform, identify: identify, chooseDefault: chooseDefault
                    )
                }
                return .unmatched(model: model, detail: unmatchedDetail(model: model, candidates: candidates))
            }
            return .family(id: family.id, model: model, source: candidates.count > 1 ? .selector : .model)
        }
        guard allowIdentify, !routing.identifiedModels.contains(model), let identification = identify(model) else {
            return .unidentified(model: model)
        }
        switch identification {
        case .managedModel(let managed):
            let resolved = resolve(
                model: managed, invocation, routing: routing, platform: platform, identify: identify,
                chooseDefault: chooseDefault, allowIdentify: false
            )
            if case .unidentified = resolved { return .unidentified(model: model) }
            return resolved
        case .family(let id):
            guard let family = routing.family(id: id) else { return .unidentified(model: model) }
            guard selectorsHold(family, invocation) else {
                return .unmatched(model: model, detail: unmatchedDetail(model: model, candidates: [family]))
            }
            return .family(id: id, model: model, source: .identified)
        }
    }

    private func resolveDefault(
        _ invocation: MereRunCommandInvocation,
        routing: MereRunCapabilityRouting,
        platform: String,
        identify: (String) -> MereRunModelIdentification?,
        chooseDefault: ([String]) -> String?
    ) -> MereRunFamilyResolution {
        guard let rule = routing.defaultModels.first(where: { rule in
            rule.applies(on: platform) && (rule.whenAny.isEmpty || rule.whenAny.contains { holds($0, invocation, family: nil) })
        }) else {
            return .unmatched(model: nil, detail: "\(command.joined(separator: " ")) has no default model on \(platform).")
        }
        if rule.family == nil, rule.models.count == 1, let model = rule.models.first, routing.identifiedModels.contains(model) {
            switch resolve(
                model: model, invocation, routing: routing, platform: platform, identify: identify,
                chooseDefault: chooseDefault, allowIdentify: true
            ) {
            case let .family(id, model, _): return .family(id: id, model: model, source: .defaultModel)
            case let other: return other
            }
        }
        var familyIDs = rule.family.map { [$0] }
            ?? Array(Set(rule.models.flatMap { model in routing.families.filter { $0.models.contains(model) }.map(\.id) }))
        if familyIDs.count > 1 {
            // A model split between families by a flag (FastH3 with and without an adapter).
            familyIDs = familyIDs.filter { id in routing.family(id: id).map { selectorsHold($0, invocation) } == true }
        }
        let family: MereRunRuntimeFamily
        let model: String?
        if familyIDs.count == 1, let only = routing.family(id: familyIDs[0]) {
            family = only
            model = rule.models.count == 1 ? rule.models.first : nil
        } else if let chosen = chooseDefault(rule.models), rule.models.contains(chosen),
                  let owner = routing.families.first(where: { $0.models.contains(chosen) }) {
            family = owner
            model = chosen
        } else {
            return .unidentified(model: rule.models.joined(separator: ", "))
        }
        guard routing.selectorsOverrideModel || selectorsHold(family, invocation) else {
            return .unmatched(model: model, detail: unmatchedDetail(model: model, candidates: [family]))
        }
        return .family(id: family.id, model: model, source: .defaultModel)
    }

    /// Selectors choose the family; the chosen family's model flag is then checked against the
    /// families that list it. An unlisted value (a local path) stays with the chosen family.
    private func resolveBySelectors(
        _ invocation: MereRunCommandInvocation,
        routing: MereRunCapabilityRouting,
        platform: String,
        identify: (String) -> MereRunModelIdentification?
    ) -> MereRunFamilyResolution {
        guard let family = routing.families.first(where: { selectorsHold($0, invocation) }) else {
            let selectors = routing.families.flatMap(\.selectors).map(\.flag)
            let passed = Array(Set(selectors)).sorted().compactMap { flag in invocation.value(flag).map { "\(flag) \($0)" } }
            return .unmatched(
                model: nil,
                detail: "No runtime of \(command.joined(separator: " ")) matches \(passed.joined(separator: ", "))."
            )
        }
        let flags = family.modelFlag.map { [$0] } ?? routing.modelFlags
        guard let model = modelValue(invocation, flags: flags) else {
            let defaultRule = routing.defaultModels.first { $0.applies(on: platform) && $0.family == family.id }
            return .family(id: family.id, model: defaultRule?.models.first, source: .selector)
        }
        let canonical: String
        if case .managedModel(let managed)? = identify(model) {
            canonical = managed
        } else {
            canonical = model
        }
        if let excluded = routing.excludedModel(id: canonical) {
            return .excluded(excluded)
        }
        if !family.models.contains(canonical), let owner = routing.families.first(where: { $0.models.contains(canonical) }) {
            return .unmatched(
                model: model,
                detail: "\(canonical) runs on \(owner.title), not \(family.title); change the model or the selector flags."
            )
        }
        return .family(id: family.id, model: canonical, source: .selector)
    }

    private func selectorsHold(_ family: MereRunRuntimeFamily, _ invocation: MereRunCommandInvocation) -> Bool {
        family.selectors.allSatisfy { holds($0, invocation, family: family.id) }
    }

    /// An omitted flag reads as the family's default for it, else the option's default. A Boolean
    /// reads as "true" when passed and "false" when omitted.
    private func holds(_ condition: MereRunFlagCondition, _ invocation: MereRunCommandInvocation, family: String?) -> Bool {
        if condition.absent { return !invocation.contains(condition.flag) }
        guard let allowed = condition.values else { return invocation.contains(condition.flag) }
        let option = options.first { $0.flag == condition.flag }
        if option?.kind == .boolean {
            return allowed.contains(String(invocation.contains(condition.flag)))
        }
        let familyDefault = option?.familyRules.first { $0.family == family }?.defaultValue
        guard let value = invocation.value(condition.flag) ?? familyDefault ?? option?.defaultValue else { return false }
        return option?.reads(value, asOneOf: allowed) ?? allowed.contains(value)
    }

    private func unmatchedDetail(model: String?, candidates: [MereRunRuntimeFamily]) -> String {
        let requirements = candidates.map { family in
            let selectors = family.selectors.map { condition in
                let rendered = arguments(satisfying: condition)
                guard let values = condition.values, rendered.count == 2 else {
                    return rendered.isEmpty ? "no \(condition.flag)" : condition.flag
                }
                return "\(condition.flag) \(values.joined(separator: "|"))"
            }
            return "\(family.title) needs \(selectors.joined(separator: " and "))"
        }
        let subject = model ?? "The default model"
        return "\(subject) can't run with these options: \(requirements.joined(separator: "; "))."
    }
}

// MARK: - Violations

extension MereRunCapabilityOption {
    /// An empty string value reads as omitted: commands treat an empty text option as not passed
    /// (`sfx generate --negative-prompt ""`), so a family that refuses the option only warns.
    /// Files, directories, numbers, and choices keep refusing an empty value.
    func readsAsOmitted(_ values: [String]) -> Bool {
        kind == .string && values.allSatisfy(\.isEmpty)
    }
}

extension MereRunCommandCapability {
    /// A named model that another family lists, when the selectors chose `family` over it: the
    /// command runs `family`'s default and the named model has no effect.
    private func overriddenModel(
        _ flag: String,
        _ invocation: MereRunCommandInvocation,
        routing: MereRunCapabilityRouting,
        family: MereRunRuntimeFamily,
        identify: (String) -> MereRunModelIdentification?
    ) -> [MereRunOptionViolation] {
        guard modelValue(invocation, flags: routing.modelFlags) == invocation.value(flag),
              let named = invocation.value(flag) else { return [] }
        let canonical: String
        if case .managedModel(let managed)? = identify(named) {
            canonical = managed
        } else {
            canonical = named
        }
        guard !family.models.contains(canonical), routing.families.contains(where: { $0.models.contains(canonical) }) else {
            return []
        }
        return [MereRunOptionViolation(
            flag: flag, kind: .valueNotAllowed(allowed: family.models), severity: .warning,
            message: "\(flag) \(named) has no effect: the other options select \(family.title)."
        )]
    }

    private func unsupported(
        _ flag: String,
        family: MereRunRuntimeFamily,
        ignored: Bool,
        usedBy: [String]
    ) -> MereRunOptionViolation {
        let appliesTo = usedBy.isEmpty ? "" : " It applies to \(Self.list(usedBy))."
        return MereRunOptionViolation(
            flag: flag,
            kind: .unsupported(supportedBy: usedBy),
            severity: ignored ? .warning : .error,
            message: ignored
                ? "\(flag) has no effect with \(family.title).\(appliesTo)"
                : "\(flag) is not supported by \(family.title).\(appliesTo)"
        )
    }

    private func ruleViolations(
        _ option: MereRunCapabilityOption,
        values: [String],
        rule: MereRunOptionFamilyRule,
        family: MereRunRuntimeFamily
    ) -> [MereRunOptionViolation] {
        let flag = option.flag
        var found: [MereRunOptionViolation] = []
        let effect = rule.severity == .warning ? "has no effect with \(family.title)" : "is not supported by \(family.title)"
        if let allowed = rule.values, let value = values.first(where: { !option.reads($0, asOneOf: allowed) }) {
            let message = allowed.count == 1
                ? "\(flag) \(value) \(effect); it runs \(allowed[0]). Remove \(flag) or pass \(allowed[0])."
                : "\(flag) \(value) \(effect); use \(Self.list(allowed, conjunction: "or"))."
            found.append(MereRunOptionViolation(
                flag: flag, kind: .valueNotAllowed(allowed: allowed), severity: rule.severity, message: message
            ))
        }
        if let range = rule.range, let value = values.first(where: { Self.outside(range, $0) }) {
            found.append(MereRunOptionViolation(
                flag: flag, kind: .outOfRange(range), severity: rule.severity,
                message: "\(flag) \(value) \(effect); use \(Self.describe(range))."
            ))
        }
        if let maximum = rule.maxCount, values.count > maximum {
            found.append(MereRunOptionViolation(
                flag: flag, kind: .tooMany(max: maximum), severity: rule.severity,
                message: "\(family.title) takes \(flag) at most \(maximum) \(maximum == 1 ? "time" : "times"); got \(values.count)."
            ))
        }
        return found
    }

    private static func outside(_ range: MereRunCapabilityRange, _ value: String) -> Bool {
        guard let number = Double(value) else { return false }
        return range.min.map { number < $0 } == true || range.max.map { number > $0 } == true
    }

    private static func describe(_ range: MereRunCapabilityRange) -> String {
        let format = { (value: Double) in value.rounded() == value ? String(Int(value)) : String(value) }
        switch (range.min, range.max) {
        case let (min?, max?) where min == max: return format(min)
        case let (min?, max?): return "a value from \(format(min)) to \(format(max))"
        case let (min?, nil): return "a value of at least \(format(min))"
        case let (nil, max?): return "a value of at most \(format(max))"
        case (nil, nil): return "a number"
        }
    }

    static func list(_ items: [String], conjunction: String = "and") -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        default: return items.dropLast().joined(separator: ", ") + " \(conjunction) " + items[items.count - 1]
        }
    }
}
