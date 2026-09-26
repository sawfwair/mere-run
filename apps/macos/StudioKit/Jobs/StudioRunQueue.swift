import Combine
import Foundation

/// Where one active job stands, in the words the run queue shows it with.
package enum StudioRunQueueStatus: Equatable {
    /// The process is working.
    case running
    /// The process launched, but the CLI is waiting for machine admission: another `mere.run`
    /// process holds the memory this one's resource class needs. It starts once that frees.
    case waitingForMemory
    /// Waiting in its lane for a free slot. `position` is 0 for the next job to start.
    case queued(position: Int)
}

/// One active job in the run queue, with where it stands and the moves its lane allows.
package struct StudioRunQueueEntry: Identifiable {
    package let job: Job
    package let status: StudioRunQueueStatus
    package let canMoveUp: Bool
    package let canMoveDown: Bool
    /// Move up is refused because the job would pass the queued pull of its model.
    package let waitsForModelDownload: Bool

    package var id: JobID { job.id }
}

/// One lane's active jobs: the running ones in start order, then the queue in the order it will
/// be admitted.
package struct StudioRunQueueSection: Identifiable {
    package let lane: JobLane
    package let entries: [StudioRunQueueEntry]
    /// How many of `entries` are waiting for a slot.
    package let queuedCount: Int

    package var id: JobLane { lane }
}

/// The run queue: everything running and waiting across every page, read straight from the
/// `JobStore`, and the actions it offers. Pure functions over the store, so grouping, positions,
/// reordering and cancellation are testable without a view.
@MainActor
package enum StudioRunQueue {
    /// The lanes whose jobs are the user's work, in the order the queue shows them. Probes are
    /// readiness checks nobody started, so they never appear.
    package static let lanes: [JobLane] = [.inference, .utility, .service]
    /// How many finished runs the queue keeps below the active ones.
    package static let recentLimit = 4

    /// Every lane that has active work, running jobs first, then the queue in admission order.
    package static func sections(in store: JobStore) -> [StudioRunQueueSection] {
        lanes.compactMap { lane in
            let queued = store.queued(in: lane)
            let running = store.running(in: lane).map { job in
                StudioRunQueueEntry(
                    job: job,
                    status: job.isAwaitingMachineAdmission ? .waitingForMemory : .running,
                    canMoveUp: false,
                    canMoveDown: false,
                    waitsForModelDownload: false
                )
            }
            let waiting = queued.enumerated().map { index, job in
                StudioRunQueueEntry(
                    job: job,
                    status: .queued(position: index),
                    canMoveUp: store.canMoveQueued(job.id, by: -1),
                    canMoveDown: store.canMoveQueued(job.id, by: 1),
                    waitsForModelDownload: store.movePassesItsDownload(job.id, by: -1)
                )
            }
            let entries = running + waiting
            return entries.isEmpty ? nil : StudioRunQueueSection(lane: lane, entries: entries, queuedCount: waiting.count)
        }
    }

    /// The most recently finished runs, newest first. Only the inference lane's: a finished
    /// `model list` or a stopped server is not a result anyone comes back for.
    package static func recentlyFinished(in store: JobStore, limit: Int = recentLimit) -> [Job] {
        Array(
            store.all
                .filter { $0.lane == .inference && $0.state.isTerminal }
                .sorted { ($0.endedAt ?? $0.submittedAt) > ($1.endedAt ?? $1.submittedAt) }
                .prefix(limit)
        )
    }

    /// Running plus queued runs: the number the Dock badge and the menu bar show. It counts the
    /// inference lane only, so background CLI reads and a resident server never make it flicker.
    package static func activeRunCount(in store: JobStore) -> Int {
        store.running(in: .inference).count + store.queued(in: .inference).count
    }

    /// The Dock badge and the menu bar count: the number while runs are active, nothing at zero.
    package static func badgeLabel(activeRuns count: Int) -> String? {
        count > 0 ? "\(count)" : nil
    }

    /// Stops a job the way its own page does: a session the way Ctrl-C does, so the CLI flushes
    /// what it has before it is terminated; anything else terminated at once. A queued job is
    /// taken out of its queue.
    package static func stop(_ job: Job, in store: JobStore) {
        if job.state.isRunning, isSession(job) {
            store.interruptThenCancel(job.id, after: StudioTaskRunner.sessionStopGrace)
        } else {
            store.cancel(job.id)
        }
    }

    /// A run of a Session task (Audio ▸ Live, Vision ▸ Live, Music ▸ Realtime): long-lived, and
    /// stopped with SIGINT so it can flush. Servers stop with SIGTERM, as the Server page stops
    /// them, and a Command Console run (`custom`) is stopped like any other command.
    package static func isSession(_ job: Job) -> Bool {
        guard job.lane == .inference, let templateID = job.request.templateID, templateID != .custom else { return false }
        return templateID.studioTask.archetype == .session
    }

    /// Takes every waiting job out of every lane's queue and leaves the running ones alone.
    /// Returns how many were removed.
    @discardableResult
    package static func cancelAllQueued(in store: JobStore) -> Int {
        let queued = lanes.flatMap { store.queued(in: $0) }
        for job in queued {
            store.cancel(job.id)
        }
        return queued.count
    }
}

// MARK: - Time left

/// Where the current progress stage began: its label, when its first determinate update arrived,
/// and how far along it already was then. The step rate since that moment is what a stage's time
/// left is measured from.
package struct StudioProgressStage: Equatable {
    package let label: String
    package let startedAt: Date
    package let startFraction: Double

    package init(label: String, startedAt: Date, startFraction: Double) {
        self.label = label
        self.startedAt = startedAt
        self.startFraction = startFraction
    }
}

/// An honest estimate of the time left in a job, and what it was measured from.
package struct StudioRunETA: Equatable {
    package enum Source: Equatable {
        /// The CLI's own download estimate (`ETA 3m 20s` on a `model pull` line).
        case download
        /// The rate of the named progress stage's steps in this run; covers that stage only.
        case stage(String)
        /// The median duration of this template and model's recent successful runs.
        case history
    }

    package let remaining: TimeInterval
    package let source: Source

    package init(remaining: TimeInterval, source: Source) {
        self.remaining = remaining
        self.source = source
    }

    /// How many recent successful runs `typicalDuration` looks at.
    package static let historySampleLimit = 5

    /// The time left in a running job, from the first of these that is measurable: the CLI's own
    /// download estimate, this run's step rate in its current stage, or how long the same
    /// template and model took recently. Nil when none is, and nil once a run has outlasted its
    /// history — an estimate is never invented to fill the space.
    package static func estimate(
        progress: StudioRunProgress?,
        stage: StudioProgressStage?,
        elapsed: TimeInterval,
        typicalDuration: TimeInterval?,
        now: Date
    ) -> StudioRunETA? {
        if let seconds = downloadSecondsLeft(progress?.detail) {
            return StudioRunETA(remaining: seconds, source: .download)
        }
        if let progress, let fraction = progress.fractionCompleted, let stage, stage.label == progress.label,
           fraction > stage.startFraction, fraction < 1 {
            let perFraction = now.timeIntervalSince(stage.startedAt) / (fraction - stage.startFraction)
            return StudioRunETA(remaining: perFraction * (1 - fraction), source: .stage(progress.label))
        }
        if let typicalDuration, typicalDuration > elapsed {
            return StudioRunETA(remaining: typicalDuration - elapsed, source: .history)
        }
        return nil
    }

    /// The median of `durations` (newest first; at most `historySampleLimit` are read), or nil
    /// with no history at all.
    package static func typicalDuration(of durations: [TimeInterval]) -> TimeInterval? {
        let sample = durations.prefix(historySampleLimit).sorted()
        guard !sample.isEmpty else { return nil }
        let middle = sample.count / 2
        return sample.count.isMultiple(of: 2) ? (sample[middle - 1] + sample[middle]) / 2 : sample[middle]
    }

    /// How long `job`'s template and model ran, newest first, across the successful runs the store
    /// still holds. Empty for work whose length says nothing about the next run: a conversation
    /// turn (its reply's length), a session or a server (they run until stopped), a raw command
    /// or a Command Console run (anything at all).
    @MainActor
    package static func recentDurations(like job: Job, in store: JobStore) -> [TimeInterval] {
        guard job.request.conversationID == nil, !StudioRunQueue.isSession(job),
              let templateID = job.request.templateID, templateID != .custom, job.lane == .inference else { return [] }
        let model = job.request.draft?.model
        return store.all.reversed().compactMap { other in
            guard other.id != job.id, other.request.templateID == templateID, other.request.draft?.model == model,
                  other.request.conversationID == nil, case .finished(0, let endedAt) = other.state,
                  let startedAt = other.startedAt else { return nil }
            return endedAt.timeIntervalSince(startedAt)
        }
    }

    /// The seconds in the CLI's download estimate: "ETA 3m 20s" → 200, "ETA 1h 12m" → 4320.
    package static func downloadSecondsLeft(_ detail: String?) -> TimeInterval? {
        guard let detail,
              let range = detail.range(of: #"ETA(\s+\d+[hms])+"#, options: .regularExpression) else { return nil }
        let units: [Character: TimeInterval] = ["h": 3_600, "m": 60, "s": 1]
        return detail[range].dropFirst(3).split(separator: " ").reduce(0) { total, part in
            guard let unit = part.last.flatMap({ units[$0] }), let amount = Double(part.dropLast()) else { return total }
            return total + amount * unit
        }
    }
}

// MARK: - Count

/// Publishes `StudioRunQueue.activeRunCount` for the Dock badge and the menu bar, and nothing else:
/// it republishes only when the number changes, so progress chatter never redraws the menu bar.
@MainActor
package final class StudioRunQueueCounter: ObservableObject {
    @Published package private(set) var activeRunCount = 0
    private var subscription: AnyCancellable?

    package init() {}

    package func attach(_ store: JobStore) {
        subscription = store.events.sink { [weak self, weak store] event in
            switch event {
            case .queued, .started, .finished, .reordered:
                guard let self, let store else { return }
                let count = StudioRunQueue.activeRunCount(in: store)
                if count != self.activeRunCount { self.activeRunCount = count }
            case .changed, .output:
                break
            }
        }
        activeRunCount = StudioRunQueue.activeRunCount(in: store)
    }
}
