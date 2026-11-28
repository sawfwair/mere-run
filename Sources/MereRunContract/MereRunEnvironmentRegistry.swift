import Foundation

/// High-level functional group for a `MERERUN_*` environment variable.
public enum MereRunEnvironmentCategory: String, Codable, Equatable, Hashable, CaseIterable, Sendable {
    case configuration
    case security
    case runtimeTuning = "runtime_tuning"
    case experimental
    case diagnostics
    case testing
    case studio
}

/// Expected value syntax for a `MERERUN_*` environment variable.
///
/// A kind describes what the variable's reader honors, not what would be convenient to
/// accept: a value that validates must behave as written at runtime.
public enum MereRunEnvironmentValueKind: Equatable, Hashable, Sendable {
    /// A switch whose reader parses `0/1`, `true/false`, `yes/no`, and `on/off`.
    case bool
    /// A switch whose reader honors only the exact values `0` and `1`.
    case flag
    case integer
    case number
    case choice([String])
    case string
    case path
    case secret

    private static let acceptedBoolTokens: Set<String> = [
        "0", "1", "true", "false", "yes", "no", "on", "off",
    ]

    /// Human-readable description of the accepted syntax.
    public var label: String {
        switch self {
        case .bool:
            return "boolean"
        case .flag:
            return "flag(0|1)"
        case .integer:
            return "integer"
        case .number:
            return "number"
        case .choice(let choices):
            return "choice(\(choices.joined(separator: "|")))"
        case .string:
            return "string"
        case .path:
            return "path"
        case .secret:
            return "secret"
        }
    }

    /// Validates a non-empty environment value. Returns `nil` when valid, or a short
    /// description of what was expected when invalid.
    ///
    /// Flags and numbers are checked untrimmed because their readers compare or parse the
    /// raw value (`== "1"`, `Int(raw)`); a stray space would silently disable them.
    public func validate(_ rawValue: String) -> String? {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        switch self {
        case .bool:
            let lowered = trimmed.lowercased()
            if Self.acceptedBoolTokens.contains(lowered) {
                return nil
            }
            return "expected a boolean (0, 1, true, false, yes, no, on, off)"
        case .flag:
            if rawValue == "0" || rawValue == "1" {
                return nil
            }
            return "expected 0 or 1"
        case .integer:
            if Int(rawValue) != nil {
                return nil
            }
            return "expected an integer"
        case .number:
            if let value = Double(rawValue), value.isFinite {
                return nil
            }
            return "expected a finite number"
        case .choice(let allowed):
            let lowered = trimmed.lowercased()
            if allowed.contains(where: { $0.lowercased() == lowered }) {
                return nil
            }
            return "expected one of: \(allowed.joined(separator: ", "))"
        case .string, .path, .secret:
            return nil
        }
    }
}

extension MereRunEnvironmentValueKind: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind
        case choices
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(String.self, forKey: .kind)
        switch kind {
        case "bool":
            self = .bool
        case "flag":
            self = .flag
        case "integer":
            self = .integer
        case "number":
            self = .number
        case "choice":
            let choices = try container.decode([String].self, forKey: .choices)
            self = .choice(choices)
        case "string":
            self = .string
        case "path":
            self = .path
        case "secret":
            self = .secret
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .kind,
                in: container,
                debugDescription: "Unknown environment value kind '\(kind)'."
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .bool:
            try container.encode("bool", forKey: .kind)
        case .flag:
            try container.encode("flag", forKey: .kind)
        case .integer:
            try container.encode("integer", forKey: .kind)
        case .number:
            try container.encode("number", forKey: .kind)
        case .choice(let choices):
            try container.encode("choice", forKey: .kind)
            try container.encode(choices, forKey: .choices)
        case .string:
            try container.encode("string", forKey: .kind)
        case .path:
            try container.encode("path", forKey: .kind)
        case .secret:
            try container.encode("secret", forKey: .kind)
        }
    }
}

/// Canonical metadata for one `MERERUN_*` environment variable.
public struct MereRunEnvironmentVariable: Codable, Equatable, Hashable, Sendable {
    public let name: String
    public let category: MereRunEnvironmentCategory
    public let valueKind: MereRunEnvironmentValueKind
    public let owner: String
    public let summary: String
    public let isSecret: Bool

    public init(
        name: String,
        category: MereRunEnvironmentCategory,
        valueKind: MereRunEnvironmentValueKind,
        owner: String,
        summary: String,
        isSecret: Bool? = nil
    ) {
        self.name = name
        self.category = category
        self.valueKind = valueKind
        self.owner = owner
        self.summary = summary
        self.isSecret = isSecret ?? (valueKind == .secret)
    }
}

/// Canonical metadata for a family of dynamically named `MERERUN_*` environment variables
/// sharing a known prefix (such as `MERERUN_SECRET_*` or `MERERUN_TEST_*`).
public struct MereRunEnvironmentPrefix: Codable, Equatable, Hashable, Sendable {
    public let prefix: String
    public let category: MereRunEnvironmentCategory
    public let valueKind: MereRunEnvironmentValueKind
    public let owner: String
    public let summary: String
    public let isSecret: Bool

    public init(
        prefix: String,
        category: MereRunEnvironmentCategory,
        valueKind: MereRunEnvironmentValueKind,
        owner: String,
        summary: String,
        isSecret: Bool? = nil
    ) {
        self.prefix = prefix
        self.category = category
        self.valueKind = valueKind
        self.owner = owner
        self.summary = summary
        self.isSecret = isSecret ?? (valueKind == .secret)
    }
}

/// One active, recognized `MERERUN_*` environment override with secret values redacted.
public struct MereRunEnvironmentOverride: Codable, Equatable, Hashable, Sendable {
    public let name: String
    public let value: String
    public let category: MereRunEnvironmentCategory
    public let owner: String
    public let isSecret: Bool

    public init(
        name: String,
        value: String,
        category: MereRunEnvironmentCategory,
        owner: String,
        isSecret: Bool
    ) {
        self.name = name
        self.value = value
        self.category = category
        self.owner = owner
        self.isSecret = isSecret
    }
}

/// One unrecognized `MERERUN_*` environment variable that is a likely typo of a registered
/// variable, with that variable as the suggestion.
public struct MereRunEnvironmentUnknownVariable: Codable, Equatable, Hashable, Sendable {
    public let name: String
    public let suggestion: String

    public init(name: String, suggestion: String) {
        self.name = name
        self.suggestion = suggestion
    }
}

/// One recognized `MERERUN_*` environment variable whose value failed syntax validation.
public struct MereRunEnvironmentInvalidValue: Codable, Equatable, Hashable, Sendable {
    public let name: String
    public let value: String
    public let expected: String

    public init(name: String, value: String, expected: String) {
        self.name = name
        self.value = value
        self.expected = expected
    }
}

/// Structured audit of all `MERERUN_*` variables present in an environment dictionary.
public struct MereRunEnvironmentAudit: Codable, Equatable, Hashable, Sendable {
    public let activeOverrides: [MereRunEnvironmentOverride]
    public let unknownVariables: [MereRunEnvironmentUnknownVariable]
    public let invalidValues: [MereRunEnvironmentInvalidValue]

    public init(
        activeOverrides: [MereRunEnvironmentOverride] = [],
        unknownVariables: [MereRunEnvironmentUnknownVariable] = [],
        invalidValues: [MereRunEnvironmentInvalidValue] = []
    ) {
        self.activeOverrides = activeOverrides
        self.unknownVariables = unknownVariables
        self.invalidValues = invalidValues
    }

    public var hasIssues: Bool {
        !unknownVariables.isEmpty || !invalidValues.isEmpty
    }

    public var isEmpty: Bool {
        activeOverrides.isEmpty && unknownVariables.isEmpty && invalidValues.isEmpty
    }

    /// Human-readable warning sentences for unknown variables and invalid values.
    public var warnings: [String] {
        var result: [String] = []
        result.reserveCapacity(unknownVariables.count + invalidValues.count)
        for unknown in unknownVariables {
            result.append(
                "Unknown environment variable \(unknown.name) (did you mean \(unknown.suggestion)?)."
            )
        }
        for invalid in invalidValues {
            result.append(
                "Invalid value '\(invalid.value)' for \(invalid.name): \(invalid.expected)."
            )
        }
        return result
    }
}

/// Canonical registry of all `MERERUN_*` environment variables and dynamic prefixes.
///
/// The CLI uses this registry to warn on misspelled variable names or invalid values at
/// startup and to surface active overrides (with secrets redacted) in `mere.run status`
/// and `/runtime/status`.
public enum MereRunEnvironmentRegistry {
    public static let prefix = "MERERUN_"
    public static let redactedPlaceholder = "[redacted]"

    /// Dynamically named `MERERUN_*` prefixes recognized across the repository.
    public static let prefixes: [MereRunEnvironmentPrefix] = [
        MereRunEnvironmentPrefix(
            prefix: "MERERUN_SECRET_",
            category: .security,
            valueKind: .secret,
            owner: "relay",
            summary: "Named secret values referenced by workflow graph nodes."
        ),
        MereRunEnvironmentPrefix(
            prefix: "MERERUN_TEST_",
            category: .testing,
            valueKind: .string,
            owner: "tests",
            summary: "Test-harness and benchmark-suite environment switches."
        ),
    ]

    /// All registered exact `MERERUN_*` variables, ordered by domain catalog.
    public static let variables: [MereRunEnvironmentVariable] = MereRunEnvironmentCatalog.allVariables

    private static let variablesByName: [String: MereRunEnvironmentVariable] = Dictionary(
        uniqueKeysWithValues: variables.map { ($0.name, $0) }
    )

    public static func variable(named name: String) -> MereRunEnvironmentVariable? {
        variablesByName[name]
    }

    public static func matchingPrefix(for name: String) -> MereRunEnvironmentPrefix? {
        prefixes.first { name.hasPrefix($0.prefix) && name.count > $0.prefix.count }
    }

    public static func isKnown(_ name: String) -> Bool {
        variablesByName[name] != nil || matchingPrefix(for: name) != nil
    }

    /// The registered value kind for an exact variable or dynamic prefix.
    public static func valueKind(for name: String) -> MereRunEnvironmentValueKind? {
        variablesByName[name]?.valueKind ?? matchingPrefix(for: name)?.valueKind
    }

    /// Returns a copy of `audit` that is safe to serve to API clients: free-form `path` and
    /// `string` values (local paths, usernames, URLs) are redacted, while switches and numbers
    /// stay visible for runtime tuning diagnostics.
    public static func redactingFreeformValues(
        _ audit: MereRunEnvironmentAudit
    ) -> MereRunEnvironmentAudit {
        func isFreeform(_ name: String) -> Bool {
            switch valueKind(for: name) {
            case .path, .string, .secret:
                return true
            case .bool, .flag, .integer, .number, .choice, nil:
                return false
            }
        }
        return MereRunEnvironmentAudit(
            activeOverrides: audit.activeOverrides.map { override in
                guard isFreeform(override.name) else { return override }
                return MereRunEnvironmentOverride(
                    name: override.name,
                    value: redactedPlaceholder,
                    category: override.category,
                    owner: override.owner,
                    isSecret: override.isSecret
                )
            },
            unknownVariables: audit.unknownVariables,
            invalidValues: audit.invalidValues.map { invalid in
                guard isFreeform(invalid.name) else { return invalid }
                return MereRunEnvironmentInvalidValue(
                    name: invalid.name,
                    value: redactedPlaceholder,
                    expected: invalid.expected
                )
            }
        )
    }

    public static func variables(in category: MereRunEnvironmentCategory) -> [MereRunEnvironmentVariable] {
        variables.filter { $0.category == category }
    }

    public static func variables(ownedBy owner: String) -> [MereRunEnvironmentVariable] {
        variables.filter { $0.owner == owner }
    }

    /// Audits every `MERERUN_*` key in `environment`, reporting active overrides (redacting
    /// secrets), likely typos of registered variables, and syntax-invalid values.
    public static func audit(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> MereRunEnvironmentAudit {
        var overrides: [MereRunEnvironmentOverride] = []
        var unknown: [MereRunEnvironmentUnknownVariable] = []
        var invalid: [MereRunEnvironmentInvalidValue] = []

        for key in environment.keys.sorted() where key.hasPrefix(prefix) && key.count > prefix.count {
            let rawValue = environment[key] ?? ""

            if let spec = variablesByName[key] {
                let displayValue = spec.isSecret ? redactedPlaceholder : rawValue
                overrides.append(
                    MereRunEnvironmentOverride(
                        name: spec.name,
                        value: displayValue,
                        category: spec.category,
                        owner: spec.owner,
                        isSecret: spec.isSecret
                    )
                )
                if let expected = spec.valueKind.validate(rawValue) {
                    invalid.append(
                        MereRunEnvironmentInvalidValue(
                            name: spec.name,
                            value: displayValue,
                            expected: expected
                        )
                    )
                }
            } else if let matchedPrefix = matchingPrefix(for: key) {
                let displayValue = matchedPrefix.isSecret ? redactedPlaceholder : rawValue
                overrides.append(
                    MereRunEnvironmentOverride(
                        name: key,
                        value: displayValue,
                        category: matchedPrefix.category,
                        owner: matchedPrefix.owner,
                        isSecret: matchedPrefix.isSecret
                    )
                )
                if let expected = matchedPrefix.valueKind.validate(rawValue) {
                    invalid.append(
                        MereRunEnvironmentInvalidValue(
                            name: key,
                            value: displayValue,
                            expected: expected
                        )
                    )
                }
            } else if let suggestion = closestMatch(for: key) {
                // Only near misses are reported: an unknown name that resembles no registered
                // variable most likely belongs to another tool (build, release, or benchmark
                // scripts), not to `mere.run`.
                unknown.append(MereRunEnvironmentUnknownVariable(name: key, suggestion: suggestion))
            }
        }

        return MereRunEnvironmentAudit(
            activeOverrides: overrides,
            unknownVariables: unknown,
            invalidValues: invalid
        )
    }

    /// Finds the closest registered `MERERUN_*` variable name using Optimal String Alignment
    /// (Damerau-Levenshtein) distance on the suffix after `MERERUN_`.
    public static func closestMatch(for unknownName: String, maxDistance: Int = 3) -> String? {
        guard maxDistance > 0 else { return nil }
        let targetSuffix = suffix(of: unknownName)
        guard !targetSuffix.isEmpty else { return nil }

        var bestName: String?
        var bestDistance = maxDistance + 1

        for candidate in variables {
            let candidateSuffix = suffix(of: candidate.name)
            let proportionalLimit = min(
                maxDistance,
                max(1, (max(candidateSuffix.count, targetSuffix.count) + 1) / 3)
            )
            // Candidates as close as the best so far stay in play so ties resolve alphabetically.
            let cutoff = min(bestDistance, proportionalLimit)
            let lengthDiff = abs(candidateSuffix.count - targetSuffix.count)
            guard lengthDiff <= cutoff else { continue }

            let distance = damerauLevenshteinDistance(targetSuffix, candidateSuffix)
            guard distance <= cutoff else { continue }
            if distance < bestDistance || bestName.map({ candidate.name < $0 }) ?? true {
                bestDistance = distance
                bestName = candidate.name
            }
        }
        return bestName
    }

    private static func suffix(of name: String) -> [Character] {
        if name.hasPrefix(prefix) {
            return Array(name.dropFirst(prefix.count))
        }
        return Array(name)
    }

    /// Optimal String Alignment distance (insertions, deletions, substitutions, adjacent
    /// transpositions each cost 1).
    static func damerauLevenshteinDistance(_ lhs: [Character], _ rhs: [Character]) -> Int {
        let m = lhs.count
        let n = rhs.count
        if m == 0 { return n }
        if n == 0 { return m }

        var prevPrev = [Int](repeating: 0, count: n + 1)
        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)

        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                let cost = lhs[i - 1] == rhs[j - 1] ? 0 : 1
                var best = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost
                )
                if i > 1, j > 1, lhs[i - 1] == rhs[j - 2], lhs[i - 2] == rhs[j - 1] {
                    best = min(best, prevPrev[j - 2] + 1)
                }
                curr[j] = best
            }
            swap(&prevPrev, &prev)
            swap(&prev, &curr)
        }
        return prev[n]
    }
}
