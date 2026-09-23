import Foundation

// Small CLI value syntaxes that Studio pages edit as controls rather than as typed strings. Each
// type owns the round trip between its structure and the exact argument the CLI parses, and states
// the CLI's own checks in the page's words, so the page never has to know the syntax.

// MARK: - Instruments

/// The instrument groups MuScriptor transcribes, as `music transcribe --list-instruments` prints
/// them: one name per line, sorted. `--instruments` takes a comma-separated subset; the CLI also
/// accepts a unique fragment of a name, but Studio always sends whole names.
package enum StudioInstrumentList {
    package static let listArguments = ["music", "transcribe", "--list-instruments"]

    /// The names the CLI printed, in its order, blank lines and stray whitespace dropped.
    package static func parse(_ output: String) -> [String] {
        var seen: Set<String> = []
        return output.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    /// "soprano_and_alto_sax" → "Soprano and alto sax".
    package static func displayName(_ name: String) -> String {
        let words = name.split(separator: "_").map(String.init)
        guard let first = words.first else { return name }
        return ([first.prefix(1).uppercased() + first.dropFirst()] + words.dropFirst()).joined(separator: " ")
    }

    /// The `--instruments` value: names in order, comma-separated; empty for automatic.
    package static func encode(_ names: [String]) -> String {
        names.joined(separator: ",")
    }

    /// The names in an `--instruments` value.
    package static func decode(_ value: String) -> [String] {
        value.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

// MARK: - Per-target LoRA ranks

/// One entry of `--lora-target-ranks`, the Klein suffix rank map: a module-name suffix and the rank
/// its adapters get, `.attn.to_q=128`. Mirrors `parseKleinTargetRankSuffixes` in `MereRunCore`.
package struct StudioTargetRank: Codable, Equatable, Identifiable {
    package var id = UUID()
    package var suffix: String
    package var rank: Int

    package init(suffix: String, rank: Int = 64) {
        self.suffix = suffix
        self.rank = rank
    }

    /// The argument: `suffix=rank` entries joined by commas; empty when there are none. A row whose
    /// suffix is still blank is left out rather than sent as `=rank`, which the CLI rejects.
    package static func encode(_ ranks: [StudioTargetRank]) -> String {
        ranks.compactMap { entry in
            let suffix = entry.suffix.trimmingCharacters(in: .whitespacesAndNewlines)
            return suffix.isEmpty ? nil : "\(suffix)=\(entry.rank)"
        }
        .joined(separator: ",")
    }

    /// The entries in an argument; a malformed entry keeps its text as the suffix with rank 0 so
    /// nothing typed is lost and the problem is reported.
    package static func decode(_ value: String) -> [StudioTargetRank] {
        value.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { entry in
                let parts = entry.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                guard parts.count == 2, let rank = Int(parts[1]) else { return StudioTargetRank(suffix: entry, rank: 0) }
                return StudioTargetRank(suffix: parts[0], rank: rank)
            }
    }

    /// The CLI's checks: every entry needs a suffix and a rank of at least 1.
    package static func problems(_ ranks: [StudioTargetRank]) -> [String] {
        ranks.enumerated().flatMap { index, entry -> [String] in
            var problems: [String] = []
            if entry.suffix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                problems.append("Target \(index + 1) needs a module suffix, like .attn.to_q.")
            }
            if entry.rank < 1 { problems.append("Target \(index + 1) needs a rank of at least 1.") }
            return problems
        }
    }
}

// MARK: - Renoise

/// `sfx generate --renoise` for Woosh models: blank for the model's default, one amount in 0…1, or
/// one amount per denoising step. Mirrors `parseRenoiseSchedule` in `MereRunCLI`. The page keeps the
/// mode it chose beside the argument, so "Per step" with nothing typed yet stays "Per step".
package enum StudioRenoise: Equatable {
    case automatic
    case amount(Double)
    /// The schedule as typed, so a token that is not a number stays visible and is reported rather
    /// than dropped.
    case schedule(String)

    package enum Mode: String, CaseIterable, Codable, Identifiable {
        case automatic
        case amount
        case schedule

        package var id: String { rawValue }

        package var title: String {
            switch self {
            case .automatic: return "Automatic"
            case .amount: return "Fixed amount"
            case .schedule: return "Per step"
            }
        }
    }

    /// The argument read in the page's chosen mode. An amount that does not parse becomes 0.5.
    package init(mode: Mode, argument: String) {
        switch mode {
        case .automatic: self = .automatic
        case .amount: self = .amount(Double(argument.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0.5)
        case .schedule: self = .schedule(argument)
        }
    }

    /// The mode an argument written elsewhere implies, for a page that has not chosen one yet:
    /// blank is automatic, one number is an amount, anything else is a schedule.
    package static func inferredMode(argument: String) -> Mode {
        let tokens = Self.tokens(argument)
        if tokens.isEmpty { return .automatic }
        if tokens.count == 1, Double(tokens[0]) != nil { return .amount }
        return .schedule
    }

    package var mode: Mode {
        switch self {
        case .automatic: return .automatic
        case .amount: return .amount
        case .schedule: return .schedule
        }
    }

    /// The `--renoise` value, always with a `.` decimal point whatever the user's locale, since the
    /// CLI reads it with `Float(_:)`; empty for automatic, which the page omits.
    package var argument: String {
        switch self {
        case .automatic: return ""
        case .amount(let amount): return CommandArguments.format(amount)
        case .schedule(let text): return Self.tokens(text).joined(separator: ",")
        }
    }

    /// The schedule's amounts, or nil while any token is not a number.
    package var scheduleValues: [Double]? {
        guard case .schedule(let text) = self else { return nil }
        let tokens = Self.tokens(text)
        let values = tokens.compactMap { Double($0) }
        return values.count == tokens.count ? values : nil
    }

    /// The CLI's checks against the run's step count.
    package func problems(steps: Int) -> [String] {
        switch self {
        case .automatic:
            return []
        case .amount(let amount):
            return (0...1).contains(amount) ? [] : ["Renoise must be between 0 and 1."]
        case .schedule(let text):
            let tokens = Self.tokens(text)
            guard !tokens.isEmpty else { return ["Enter one renoise amount per step, separated by commas."] }
            guard let values = scheduleValues else {
                return ["Renoise amounts must be numbers separated by commas, with a point for decimals."]
            }
            var problems: [String] = []
            if values.count != steps {
                problems.append("The renoise schedule has \(values.count) values but the run has \(steps) steps.")
            }
            if values.contains(where: { !(0...1).contains($0) }) { problems.append("Renoise values must be between 0 and 1.") }
            return problems
        }
    }

    private static func tokens(_ text: String) -> [String] {
        text.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }
}

