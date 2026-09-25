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
    public func resolveFamily(
        _ invocation: MereRunCommandInvocation,
        platform: String = "macos",
        identify: (String) -> MereRunModelIdentification? = { _ in nil }
    ) -> MereRunFamilyResolution {
        guard let routing else { return .unrouted }
        if routing.routesBySelectors {
            return resolveBySelectors(invocation, routing: routing, platform: platform, identify: identify)
        }
        guard let model = modelValue(invocation, flags: routing.modelFlags) else {
            return resolveDefault(invocation, routing: routing, platform: platform)
        }
        return resolve(model: model, invocation, routing: routing, identify: identify, allowIdentify: true)
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
                familyRules: rule.map { [$0] } ?? []
            )
        }
    }

    /// Every way `invocation` leaves `family`'s scope, errors and warnings, in option order.
    public func violations(_ invocation: MereRunCommandInvocation, family: String) -> [MereRunOptionViolation] {
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
            if let families = option.families, !families.contains(family) {
                let titles = families.compactMap { routing.family(id: $0)?.title }
                let ignored = option.ignoredBy.contains(family) || option.readsAsOmitted(values)
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
        identify: (String) -> MereRunModelIdentification? = { _ in nil }
    ) -> MereRunFamilyResolutionReport {
        let resolution = resolveFamily(invocation, platform: platform, identify: identify)
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
            let found = violations(invocation, family: familyID)
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
    /// The shortest arguments that make `condition` hold: the flag and its first allowed value,
    /// the flag alone for a presence condition or a Boolean that must be on, and nothing for a
    /// Boolean that must be off.
    public func arguments(satisfying condition: MereRunFlagCondition) -> [String] {
        guard let value = condition.values?.first else { return [condition.flag] }
        guard options.first(where: { $0.flag == condition.flag })?.kind == .boolean else {
            return [condition.flag, value]
        }
        return value == "true" ? [condition.flag] : []
    }

    private func modelValue(_ invocation: MereRunCommandInvocation, flags: [String]) -> String? {
        flags.lazy.compactMap { invocation.value($0) }.first { !$0.isEmpty }
    }

    private func resolve(
        model: String,
        _ invocation: MereRunCommandInvocation,
        routing: MereRunCapabilityRouting,
        identify: (String) -> MereRunModelIdentification?,
        allowIdentify: Bool
    ) -> MereRunFamilyResolution {
        if let excluded = routing.excludedModel(id: model) {
            return .excluded(excluded)
        }
        let candidates = routing.families.filter { $0.models.contains(model) }
        if !candidates.isEmpty {
            let matching = candidates.filter { selectorsHold($0, invocation) }
            guard let family = matching.first else {
                return .unmatched(model: model, detail: unmatchedDetail(model: model, candidates: candidates))
            }
            return .family(id: family.id, model: model, source: candidates.count > 1 ? .selector : .model)
        }
        guard allowIdentify, let identification = identify(model) else {
            return .unidentified(model: model)
        }
        switch identification {
        case .managedModel(let managed):
            switch resolve(model: managed, invocation, routing: routing, identify: identify, allowIdentify: false) {
            case .family(let id, _, let source):
                return .family(id: id, model: managed, source: source)
            case .unidentified:
                return .unidentified(model: model)
            case let other:
                return other
            }
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
        platform: String
    ) -> MereRunFamilyResolution {
        guard let rule = routing.defaultModels.first(where: { rule in
            rule.applies(on: platform) && (rule.whenAny.isEmpty || rule.whenAny.contains { holds($0, invocation, family: nil) })
        }) else {
            return .unmatched(model: nil, detail: "\(command.joined(separator: " ")) has no default model on \(platform).")
        }
        let familyIDs = rule.family.map { [$0] }
            ?? Array(Set(rule.models.flatMap { model in routing.families.filter { $0.models.contains(model) }.map(\.id) }))
        guard familyIDs.count == 1, let family = routing.family(id: familyIDs[0]) else {
            return .unidentified(model: rule.models.joined(separator: ", "))
        }
        let model = rule.models.count == 1 ? rule.models.first : nil
        guard selectorsHold(family, invocation) else {
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
        guard let allowed = condition.values else { return invocation.contains(condition.flag) }
        let option = options.first { $0.flag == condition.flag }
        if option?.kind == .boolean {
            return allowed.contains(String(invocation.contains(condition.flag)))
        }
        let familyDefault = option?.familyRules.first { $0.family == family }?.defaultValue
        guard let value = invocation.value(condition.flag) ?? familyDefault ?? option?.defaultValue else { return false }
        return allowed.contains(value)
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
        if let allowed = rule.values, let value = values.first(where: { !Self.allows(allowed, $0, kind: option.kind) }) {
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

    /// A number matches an allowed value numerically, so `--guidance 0.0` meets a rule of `0`.
    private static func allows(_ allowed: [String], _ value: String, kind: MereRunCapabilityValueKind) -> Bool {
        guard kind == .integer || kind == .number, let number = Double(value) else {
            return allowed.contains(value)
        }
        return allowed.contains { Double($0) == number }
    }

    private static func outside(_ range: MereRunCapabilityRange, _ value: String) -> Bool {
        guard let number = Double(value) else { return false }
        return range.min.map { number < $0 } == true || range.max.map { number > $0 } == true
    }

    private static func describe(_ range: MereRunCapabilityRange) -> String {
        let format = { (value: Double) in value.rounded() == value ? String(Int(value)) : String(value) }
        switch (range.min, range.max) {
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
