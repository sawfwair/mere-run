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

    /// The argument: `suffix=rank` entries joined by commas; empty when there are none.
    package static func encode(_ ranks: [StudioTargetRank]) -> String {
        ranks.map { "\($0.suffix.trimmingCharacters(in: .whitespacesAndNewlines))=\($0.rank)" }.joined(separator: ",")
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
/// one amount per denoising step. Mirrors `parseRenoiseSchedule` in `MereRunCLI`.
package enum StudioRenoise: Equatable {
    case automatic
    case amount(Double)
    case schedule([Double])

    package enum Mode: String, CaseIterable, Identifiable {
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

    /// Reads the argument as the CLI would; text it cannot read becomes an empty schedule so the
    /// page shows the problem instead of silently dropping the value.
    package init(argument: String) {
        let values = argument.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        if values.isEmpty {
            self = .automatic
        } else if values.count == 1, let amount = Double(values[0]) {
            self = .amount(amount)
        } else {
            self = .schedule(values.compactMap(Double.init))
        }
    }

    package var mode: Mode {
        switch self {
        case .automatic: return .automatic
        case .amount: return .amount
        case .schedule: return .schedule
        }
    }

    /// The `--renoise` value; empty for automatic, which the page omits.
    package var argument: String {
        switch self {
        case .automatic: return ""
        case .amount(let amount): return Self.format(amount)
        case .schedule(let values): return values.map(Self.format).joined(separator: ",")
        }
    }

    /// The CLI's checks against the run's step count.
    package func problems(steps: Int) -> [String] {
        switch self {
        case .automatic:
            return []
        case .amount(let amount):
            return (0...1).contains(amount) ? [] : ["Renoise must be between 0 and 1."]
        case .schedule(let values):
            var problems: [String] = []
            if values.isEmpty {
                problems.append("Enter one renoise amount per step, separated by commas.")
            } else if values.count != steps {
                problems.append("The renoise schedule has \(values.count) values but the run has \(steps) steps.")
            }
            if values.contains(where: { !(0...1).contains($0) }) { problems.append("Renoise values must be between 0 and 1.") }
            return problems
        }
    }

    private static func format(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(0...4)).grouping(.never))
    }
}
