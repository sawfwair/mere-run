import Foundation

/// How a capability chooses the code path that runs it. Every capability that loads a model
/// declares one, including single-family ones: the family list is also what model pickers offer
/// and what the managed-model coverage test checks.
public struct MereRunCapabilityRouting: Codable, Equatable, Sendable {
    /// Options whose value names the model, highest precedence first: `["--model-root", "--model"]`
    /// for video generate, `["--model"]` for text chat (`--model-root` there locates weights but
    /// never changes the family). Empty when families are chosen by selectors alone.
    public let modelFlags: [String]
    /// What runs when every model flag is omitted. The first rule whose `whenAny` holds (or that
    /// has none) and whose `platforms` include the platform wins.
    public let defaultModels: [MereRunDefaultModelRule]
    public let families: [MereRunRuntimeFamily]
    /// Managed models that list this command, or that its pickers would otherwise offer, but
    /// that cannot run it. Resolving to one is an error that names `reason`.
    public let excludedModels: [MereRunExcludedModel]
    /// True when what is installed can change the family a managed id runs: the command loads a
    /// root an environment override or a local layout names instead of the id's own install
    /// (`music generate` and `MERERUN_MUSIC_ACESTEP_ROOT`). The resolver then asks `identify`
    /// about a listed or default model first, and uses the id's family only when it can't tell.
    /// Shells without an identifier resolve the id as listed.
    public let identifiesInstalledModels: Bool

    enum CodingKeys: String, CodingKey {
        case modelFlags = "model_flags"
        case defaultModels = "default_models"
        case families
        case excludedModels = "excluded_models"
        case identifiesInstalledModels = "identifies_installed_models"
    }

    public init(
        modelFlags: [String],
        defaultModels: [MereRunDefaultModelRule] = [],
        families: [MereRunRuntimeFamily],
        excludedModels: [MereRunExcludedModel] = [],
        identifiesInstalledModels: Bool = false
    ) {
        self.modelFlags = modelFlags
        self.defaultModels = defaultModels
        self.families = families
        self.excludedModels = excludedModels
        self.identifiesInstalledModels = identifiesInstalledModels
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelFlags = try container.decode([String].self, forKey: .modelFlags)
        defaultModels = try container.decode([MereRunDefaultModelRule].self, forKey: .defaultModels)
        families = try container.decode([MereRunRuntimeFamily].self, forKey: .families)
        excludedModels = try container.decode([MereRunExcludedModel].self, forKey: .excludedModels)
        identifiesInstalledModels = try container.decodeIfPresent(Bool.self, forKey: .identifiesInstalledModels) ?? false
    }

    /// `identifies_installed_models` is written only when true.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelFlags, forKey: .modelFlags)
        try container.encode(defaultModels, forKey: .defaultModels)
        try container.encode(families, forKey: .families)
        try container.encode(excludedModels, forKey: .excludedModels)
        if identifiesInstalledModels { try container.encode(true, forKey: .identifiesInstalledModels) }
    }

    public func family(id: String) -> MereRunRuntimeFamily? {
        families.first { $0.id == id }
    }

    public func excludedModel(id: String) -> MereRunExcludedModel? {
        excludedModels.first { $0.id == id }
    }

    /// True when a flag's value, not the model, picks the family (vision.ocr `--backend`, or a
    /// capability with no model flag). The model flag of the chosen family is then only checked.
    public var routesBySelectors: Bool {
        modelFlags.isEmpty || families.contains { $0.modelFlag != nil }
    }
}

/// One code path in the CLI with a fixed option surface.
public struct MereRunRuntimeFamily: Codable, Equatable, Sendable {
    /// Stable kebab-case id, unique within the capability: "ltx25-full", "wan22-ti2v".
    public let id: String
    /// What shells and error messages call it: "LTX-2.5 Full".
    public let title: String
    /// Managed model ids that select this family. Exact ids; upstream-repository aliases are
    /// normalized by the CLI's identifier, never listed here.
    public let models: [String]
    /// Flag values that must all hold for this family (speech.transcribe `--backend qwen`,
    /// speech.synthesize `--mode clone`). Empty: the model alone decides.
    public let selectors: [MereRunFlagCondition]
    /// The option that carries this family's model when it is not one of `routing.modelFlags`
    /// (vision.ocr Infinity takes `--infinity-model`). `nil`: `routing.modelFlags`.
    public let modelFlag: String?

    enum CodingKeys: String, CodingKey {
        case id, title, models, selectors
        case modelFlag = "model_flag"
    }

    public init(
        id: String,
        title: String,
        models: [String],
        selectors: [MereRunFlagCondition] = [],
        modelFlag: String? = nil
    ) {
        self.id = id
        self.title = title
        self.models = models
        self.selectors = selectors
        self.modelFlag = modelFlag
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        title = try container.decode(String.self, forKey: .title)
        models = try container.decode([String].self, forKey: .models)
        selectors = try container.decodeIfPresent([MereRunFlagCondition].self, forKey: .selectors) ?? []
        modelFlag = try container.decodeIfPresent(String.self, forKey: .modelFlag)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(title, forKey: .title)
        try container.encode(models, forKey: .models)
        if !selectors.isEmpty { try container.encode(selectors, forKey: .selectors) }
        try container.encodeIfPresent(modelFlag, forKey: .modelFlag)
    }
}

/// A condition on one flag of the same capability.
public struct MereRunFlagCondition: Codable, Equatable, Sendable {
    public let flag: String
    /// `nil`: the flag is present (a Boolean is on). Otherwise the flag's value, or its default
    /// when omitted, is one of these, rendered as the CLI parses them.
    public let values: [String]?

    public init(flag: String, values: [String]? = nil) {
        self.flag = flag
        self.values = values
    }
}

public struct MereRunDefaultModelRule: Codable, Equatable, Sendable {
    /// Any one of these holding selects the rule; empty means always.
    public let whenAny: [MereRunFlagCondition]
    /// The model, or the candidates the CLI chooses between by machine (text chat picks Gemma 4
    /// 12B 4-bit or Nano by memory). Empty when the default is not a managed model.
    public let models: [String]
    /// The family that runs the default when `models` does not name it: a default the CLI
    /// downloads by repository (vision caption), or candidates that span families. `nil`: the
    /// family whose `models` list every candidate.
    public let family: String?
    /// `nil`: every platform. Otherwise `["macos"]` or `["linux"]`.
    public let platforms: [String]?

    enum CodingKeys: String, CodingKey {
        case whenAny = "when_any", models, family, platforms
    }

    public init(
        whenAny: [MereRunFlagCondition] = [],
        models: [String],
        family: String? = nil,
        platforms: [String]? = nil
    ) {
        self.whenAny = whenAny
        self.models = models
        self.family = family
        self.platforms = platforms
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        whenAny = try container.decodeIfPresent([MereRunFlagCondition].self, forKey: .whenAny) ?? []
        models = try container.decode([String].self, forKey: .models)
        family = try container.decodeIfPresent(String.self, forKey: .family)
        platforms = try container.decodeIfPresent([String].self, forKey: .platforms)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        if !whenAny.isEmpty { try container.encode(whenAny, forKey: .whenAny) }
        try container.encode(models, forKey: .models)
        try container.encodeIfPresent(family, forKey: .family)
        try container.encodeIfPresent(platforms, forKey: .platforms)
    }

    public func applies(on platform: String) -> Bool {
        platforms?.contains(platform) ?? true
    }
}

public struct MereRunExcludedModel: Codable, Equatable, Sendable {
    public let id: String
    /// One sentence naming the command it does run: "Woosh CLAP scores audio; use `sfx clap score`."
    public let reason: String

    public init(id: String, reason: String) {
        self.id = id
        self.reason = reason
    }
}

/// How an option is narrowed for one family that uses it. A rule may also name a family in the
/// option's `ignored_by`, with only `values` or a `range`: that family runs without the option but
/// refuses any value except these, as the CLI's value-based checks let only the value it runs with
/// through (`--vocal-language en` on YuE2, `--sample-steps 8` on Krea 2). A tolerated value warns
/// that the option has no effect; any other fails as if the family rejected the option.
public struct MereRunOptionFamilyRule: Codable, Equatable, Sendable {
    public let family: String
    /// Allowed values, rendered as the CLI parses them. One value means the family fixes it. For a
    /// `.choice` option this is a subset of `choices`; for a numeric option, the exact values the
    /// family accepts (music separate `--overlap`, FastH3 `--steps 5`).
    public let values: [String]?
    /// What the family runs with when the option is omitted, when that differs from the option's
    /// `default_value` (music `--quality song` on ACE only; `--steps` per family).
    public let defaultValue: String?
    /// The family's own numeric bounds, replacing the option's `range`.
    public let range: MereRunCapabilityRange?
    /// The family cannot run without it (H3 Ref2VA `--reference`, Wan `--image`).
    public let required: Bool
    /// Most occurrences of a repeatable option (Z-Image `--lora` 1, Qwen-Edit `--ref-image` 3).
    public let maxCount: Int?
    /// What a value outside `values` or `range`, or past `maxCount`, does today: `.error` when the
    /// CLI rejects it, `.warning` when the CLI accepts it and runs its own value instead. A
    /// missing required option is always an error.
    public let severity: MereRunOptionViolation.Severity

    enum CodingKeys: String, CodingKey {
        case family, values, range, required, severity
        case defaultValue = "default_value"
        case maxCount = "max_count"
    }

    public init(
        family: String,
        values: [String]? = nil,
        defaultValue: String? = nil,
        range: MereRunCapabilityRange? = nil,
        required: Bool = false,
        maxCount: Int? = nil,
        severity: MereRunOptionViolation.Severity = .error
    ) {
        self.family = family
        self.values = values
        self.defaultValue = defaultValue
        self.range = range
        self.required = required
        self.maxCount = maxCount
        self.severity = severity
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        family = try container.decode(String.self, forKey: .family)
        values = try container.decodeIfPresent([String].self, forKey: .values)
        defaultValue = try container.decodeIfPresent(String.self, forKey: .defaultValue)
        range = try container.decodeIfPresent(MereRunCapabilityRange.self, forKey: .range)
        required = try container.decodeIfPresent(Bool.self, forKey: .required) ?? false
        maxCount = try container.decodeIfPresent(Int.self, forKey: .maxCount)
        severity = try container.decodeIfPresent(MereRunOptionViolation.Severity.self, forKey: .severity) ?? .error
    }

    /// `required` and an `.error` severity are the common case and stay absent.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(family, forKey: .family)
        try container.encodeIfPresent(values, forKey: .values)
        try container.encodeIfPresent(defaultValue, forKey: .defaultValue)
        try container.encodeIfPresent(range, forKey: .range)
        if required { try container.encode(true, forKey: .required) }
        try container.encodeIfPresent(maxCount, forKey: .maxCount)
        if severity != .error { try container.encode(severity, forKey: .severity) }
    }
}

// MARK: - Typed authoring

/// A capability's family ids, declared as an enum per capability so a typo in a scope or rule is
/// a compile error rather than a silent miss.
protocol MereRunFamilyID: RawRepresentable, CaseIterable, Hashable where RawValue == String {}

extension MereRunFamilyID {
    /// Only `used` use the option; `ignoredBy` accept it without effect; the rest reject it.
    static func only(_ used: Self..., ignoredBy: [Self] = []) -> MereRunOptionScope<Self> {
        MereRunOptionScope(families: used, ignoredBy: ignoredBy)
    }

    /// Every family uses the option except `rejecting`, which rejects it, and `ignoredBy`.
    static func except(_ rejecting: Self..., ignoredBy: [Self] = []) -> MereRunOptionScope<Self> {
        let excluded = Set(rejecting).union(ignoredBy)
        return MereRunOptionScope(families: allCases.filter { !excluded.contains($0) }, ignoredBy: ignoredBy)
    }

    static func rule(
        _ family: Self,
        values: [String]? = nil,
        defaultValue: String? = nil,
        range: MereRunCapabilityRange? = nil,
        required: Bool = false,
        maxCount: Int? = nil,
        severity: MereRunOptionViolation.Severity = .error
    ) -> MereRunOptionRule<Self> {
        .rule(family, values: values, defaultValue: defaultValue, range: range, required: required,
              maxCount: maxCount, severity: severity)
    }
}

extension MereRunRuntimeFamily {
    init<Family: MereRunFamilyID>(
        _ id: Family,
        title: String,
        models: [String],
        selectors: [MereRunFlagCondition] = [],
        modelFlag: String? = nil
    ) {
        self.init(id: id.rawValue, title: title, models: models, selectors: selectors, modelFlag: modelFlag)
    }
}

extension MereRunDefaultModelRule {
    /// The unconditional default: one model, or the candidates the CLI picks between by machine.
    static func always(_ models: String...) -> Self {
        Self(models: models)
    }

    /// A default the CLI downloads by repository rather than by managed id.
    static func unmanaged<Family: MereRunFamilyID>(_ family: Family) -> Self {
        Self(models: [], family: family.rawValue)
    }
}

extension Array where Element == MereRunExcludedModel {
    /// Several models excluded for the same reason: `.models([...], reason: ...).and([...], reason: ...)`.
    static func models(_ ids: [String], reason: String) -> Self {
        ids.map { MereRunExcludedModel(id: $0, reason: reason) }
    }

    func and(_ ids: [String], reason: String) -> Self {
        self + .models(ids, reason: reason)
    }
}

/// Which families use an option, and which accept and ignore it. Every other family rejects it.
/// Written from the family enum so the generic resolves: `VideoFamily.only(.wan, .ltx25Full)`.
struct MereRunOptionScope<Family: MereRunFamilyID> {
    let families: [Family]
    let ignoredBy: [Family]
}

/// One family's narrowing of an option. The first rule of a list is written from the family
/// enum (`VideoFamily.rule(.fastH3, values: ["5"])`); the rest can use `.rule(...)`.
struct MereRunOptionRule<Family: MereRunFamilyID> {
    let rule: MereRunOptionFamilyRule

    static func rule(
        _ family: Family,
        values: [String]? = nil,
        defaultValue: String? = nil,
        range: MereRunCapabilityRange? = nil,
        required: Bool = false,
        maxCount: Int? = nil,
        severity: MereRunOptionViolation.Severity = .error
    ) -> Self {
        Self(rule: MereRunOptionFamilyRule(
            family: family.rawValue, values: values, defaultValue: defaultValue, range: range,
            required: required, maxCount: maxCount, severity: severity
        ))
    }
}

extension MereRunCapabilityOption {
    /// This option with its family scope and rules: `.scoped(F.only(.wan, .ltx25Full))`,
    /// `.scoped(F.except(.fastH3), .rule(.fastH3, values: ["5"]))`, where `F` is the capability's
    /// family enum.
    func scoped<Family: MereRunFamilyID>(
        _ scope: MereRunOptionScope<Family>,
        _ rules: MereRunOptionRule<Family>...
    ) -> Self {
        // Declaration order keeps the serialized lists stable however a scope was written.
        let sorted = { (families: [Family]) in Family.allCases.filter(families.contains).map(\.rawValue) }
        return with(families: sorted(scope.families), ignoredBy: sorted(scope.ignoredBy), rules: rules.map(\.rule))
    }

    /// This option with rules for families that use it; every family still uses it.
    func scoped<Family: MereRunFamilyID>(_ rules: MereRunOptionRule<Family>...) -> Self {
        with(families: families, ignoredBy: ignoredBy, rules: rules.map(\.rule))
    }

    private func with(families: [String]?, ignoredBy: [String], rules: [MereRunOptionFamilyRule]) -> Self {
        Self(flag: flag, aliases: aliases, label: label, kind: kind, required: required, repeatable: repeatable,
            choices: choices, defaultValue: defaultValue, group: group, tier: tier, range: range,
            dependsOn: dependsOn, families: families, ignoredBy: ignoredBy, familyRules: rules,
            choiceSpellings: choiceSpellings)
    }
}
