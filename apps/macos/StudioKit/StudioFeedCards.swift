import Foundation

/// What one Library row of the current mode looks like in the feed: a finished generation with
/// its outputs, a run in flight, a run waiting for a lane, or a failure. Derived from the row
/// and, while the run is alive, its `Job`; the views observe the `Job` directly for progress.
package enum StudioFeedCardKind: Equatable {
    /// Finished with exit 0: the prompt, its chips, and every output.
    case generation
    /// The process is alive; `job` carries progress, log, and Cancel.
    case running
    /// Submitted and waiting for a free inference slot (or a stale queued row with no job).
    case queued
    /// Exited non-zero, was cancelled, or failed preflight.
    case failed
}

package struct StudioFeedCard: Identifiable, Equatable {
    package let item: StudioLibraryItem
    package let kind: StudioFeedCardKind
    /// The row's job while the store still retains it; nil for rows from earlier sessions.
    package let job: Job?

    package var id: UUID { item.id }

    package static func == (lhs: StudioFeedCard, rhs: StudioFeedCard) -> Bool {
        lhs.item == rhs.item && lhs.kind == rhs.kind && lhs.job === rhs.job
    }
}

/// Builds the feed from the Library: the mode's non-conversation rows, oldest first so the
/// newest generation sits just above the composer.
@MainActor
package enum StudioFeedCardBuilder {
    package static func cards(
        items: [StudioLibraryItem],
        mode: StudioMode,
        job: (UUID) -> Job?
    ) -> [StudioFeedCard] {
        items
            .filter { $0.mode == mode && !$0.isConversation }
            .sorted { $0.createdAt < $1.createdAt }
            .map { item in
                let job = job(item.id)
                return StudioFeedCard(item: item, kind: kind(for: item, job: job), job: job)
            }
    }

    /// The job's state wins while it is alive; the Library row decides once it is gone or done.
    package static func kind(for item: StudioLibraryItem, job: Job?) -> StudioFeedCardKind {
        if let job {
            switch job.state {
            case .queued: return .queued
            case .running: return .running
            case .finished(let exit, _): return exit == 0 ? .generation : .failed
            case .cancelled, .preflightFailed: return .failed
            }
        }
        switch item.status {
        case .completed: return .generation
        case .running: return .running
        case .queued: return .queued
        case .failed, .cancelled, .interrupted: return .failed
        }
    }

    /// The queued cards in submission order; the first is "next".
    package static func queuePosition(of card: StudioFeedCard, in cards: [StudioFeedCard]) -> Int? {
        let queued = cards.filter { $0.kind == .queued }
        return queued.firstIndex(where: { $0.id == card.id })
    }
}

/// The one line a failure card leads with: the last meaningful line of what the run wrote,
/// never the whole stdout/stderr dump.
package enum StudioFailureSummary {
    /// Lines that carry no diagnosis on their own: progress echoes, the shell's exit note,
    /// tracebacks' framing, blank separators.
    private static let noisePrefixes = [
        "traceback", "file \"", "  ", "^", "exited with code", "completed with exit code",
        "termination requested", "stderr", "warning:", "{", "[", "generating (", "denoising ",
    ]

    /// Summarizes a run's captured text (`StudioLibraryItem.outputText`, which is the stdout and
    /// stderr the run left behind) or its log lines, newest last.
    package static func summary(outputText: String?, logLines: [String] = [], exitCode: Int32?) -> String {
        // The log is newer than the captured text, so it is searched first (from its end).
        let candidates = (outputText ?? "").components(separatedBy: .newlines) + logLines
        if let line = candidates.reversed().first(where: isMeaningful) {
            return cleaned(line)
        }
        if let exitCode {
            switch exitCode {
            case JobResult.cancelledBeforeStartExitCode: return "Cancelled."
            case 64: return "The request was invalid."
            default: return "The run exited with code \(exitCode)."
            }
        }
        return "The run failed."
    }

    package static func isMeaningful(_ rawLine: String) -> Bool {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        guard line.count > 3 else { return false }
        let lower = line.lowercased()
        if noisePrefixes.contains(where: { lower.hasPrefix($0) }) { return false }
        if StudioProgressParser.parse(line) != nil { return false }
        return true
    }

    /// Drops CLI framing ("error:", "Error:", "mere.run:") and trailing punctuation noise.
    private static func cleaned(_ rawLine: String) -> String {
        var line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["mere.run:", "error:", "Error:", "ERROR:", "fatal:"] where line.hasPrefix(prefix) {
            line = String(line.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
        }
        if let first = line.first, first.isLowercase {
            line = first.uppercased() + line.dropFirst()
        }
        return line
    }
}

/// The compact status a running card shows beside its bar: "Denoising 15/24" from a step
/// progress, the label alone otherwise.
package enum StudioRunningStatus {
    package static func text(progress: StudioRunProgress?, fallback: String) -> String {
        guard let progress else { return fallback }
        if let detail = progress.detail, let ratio = stepRatio(detail) {
            return "\(progress.label) \(ratio)"
        }
        if let detail = progress.detail, !detail.isEmpty {
            return "\(progress.label) · \(detail)"
        }
        return progress.label
    }

    /// "Step 15 of 24" → "15/24".
    private static func stepRatio(_ detail: String) -> String? {
        let words = detail.split(separator: " ")
        guard words.count >= 4, words[0] == "Step", words[2] == "of",
              Int(words[1]) != nil, Int(words[3]) != nil else { return nil }
        return "\(words[1])/\(words[3])"
    }
}
