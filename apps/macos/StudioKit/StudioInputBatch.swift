import Foundation
import UniformTypeIdentifiers

// Batch inputs: several files given to a task whose input takes one run as many times, once per
// file. Which slots batch is read from the slot schema — never a per-page table — and the batch
// lives in the draft (`batchInputPaths`), so drops, pastes, picks, removals, and ⌘Z all go through
// the draft's one setter. Each file becomes its own run through `StudioTaskRunner`, named by
// `StudioOutputLocation` like any other, and the Library rows of one Run share a `batchGroup`.

// MARK: - Which slots batch

extension StudioAttachmentSlot {
    /// Whether `slot`, the first of `task`'s well, runs once per file when given several: it holds
    /// one file (not a list, which takes several natively, and not a folder), the run cannot
    /// happen without it, it stays in the well between runs (not Chat's per-turn picture), and
    /// the task makes one result per run on a Generate or Analyze surface. A live session, a
    /// training project, a Manage page, and a conversation never batch.
    package func batchesRuns(for task: StudioTask) -> Bool {
        !allowsMultiple && isRequired && !isTransient && !acceptedTypes.contains(.folder)
            && [.generate, .analyze].contains(task.archetype)
            && task.mode?.isConversational != true
    }

    /// Whether `draft` holds a batch in this slot: two files or more.
    package func isBatched<Draft: StudioAttachmentDraft>(in draft: Draft) -> Bool {
        batches && draft.batchInputPaths.count > 1
    }

    /// The files a Run takes through this slot, one run each: the batch, else the slot's file.
    package func runPaths<Draft: StudioAttachmentDraft>(in draft: Draft) -> [String] {
        isBatched(in: draft) ? draft.batchInputPaths : paths(in: draft)
    }

    /// Makes `paths` the slot's files, in order and without repeats: two or more are a batch whose
    /// first file is also the slot's (so the canvas and the Command view show the first run), one
    /// is a plain attachment, none clears the slot.
    package func setBatch<Draft: StudioAttachmentDraft>(_ paths: [String], in draft: inout Draft) {
        var seen = Set<String>()
        let unique = paths.filter { seen.insert($0).inserted }
        let first = unique.first ?? ""
        if draft.attachmentText(for: storage) != first { draft.setAttachmentText(first, for: storage) }
        draft.batchInputPaths = unique.count > 1 ? unique : []
    }

    /// Takes one file out of the batch; a batch of two becomes the other file on its own.
    package func removeFromBatch<Draft: StudioAttachmentDraft>(_ path: String, in draft: inout Draft) {
        setBatch(runPaths(in: draft).filter { $0 != path }, in: &draft)
    }
}

extension Array where Element == StudioAttachmentSlot {
    /// The slots with the first marked as batching when `batchesRuns(for:)` says it does.
    package func markingBatchInput(for task: StudioTask) -> [StudioAttachmentSlot] {
        guard var primary = first, primary.batchesRuns(for: task) else { return self }
        primary.batches = true
        return [primary] + dropFirst()
    }

    /// How many runs Run makes: the batch's size while the batching slot holds one, else nil.
    package func batchRunCount<Draft: StudioAttachmentDraft>(in draft: Draft) -> Int? {
        guard let slot = first(where: \.batches), slot.isBatched(in: draft) else { return nil }
        return draft.batchInputPaths.count
    }
}

extension StudioAttachmentDraft {
    /// This draft as one run of its batch: `path` in `slot`, and no batch.
    package func running(_ path: String, in slot: StudioAttachmentSlot) -> Self {
        var single = self
        single.batchInputPaths = []
        single.setAttachmentText(path, for: slot.storage)
        return single
    }
}

// MARK: - Checking a batch before it runs

/// One file of a batch as the check before submitting found it.
package struct StudioBatchFileCheck: Equatable, Identifiable {
    package let path: String
    /// Why this file cannot run, in the banner's words; nil when it can.
    package let problem: String?

    package init(path: String, problem: String?) {
        self.path = path
        self.problem = problem
    }

    package var id: String { path }
    package var fileName: String { URL(fileURLWithPath: path).lastPathComponent }
}

/// Every file of a batch, checked up front: readiness once for the batch, then each file on its
/// own, so a bad file is named before anything is submitted.
package struct StudioBatchReview: Equatable {
    package let files: [StudioBatchFileCheck]

    package init(files: [StudioBatchFileCheck]) {
        self.files = files
    }

    package var runnable: [String] { files.filter { $0.problem == nil }.map(\.path) }
    package var rejected: [StudioBatchFileCheck] { files.filter { $0.problem != nil } }

    /// What Run does with the review.
    package enum Decision: Equatable {
        /// Every file can run.
        case runAll
        /// Some files cannot run; ask before running the rest.
        case confirmSkipping
        /// No file can run; say why and submit nothing.
        case refuse(String)
    }

    package var decision: Decision {
        let rejected = rejected
        if rejected.isEmpty { return .runAll }
        if rejected.count < files.count { return .confirmSkipping }
        let reasons = Set(rejected.compactMap(\.problem))
        if reasons.count == 1, let reason = reasons.first {
            return .refuse(files.count == 2 ? "Neither file can run. \(reason)" : "None of the \(files.count) files can run. \(reason)")
        }
        return .refuse("None of the \(files.count) files can run. \(rejected[0].fileName): \(rejected[0].problem ?? "")")
    }

    /// The confirmation's title: "2 of 12 files can't run".
    package var confirmationTitle: String {
        let count = rejected.count
        return count == 1 ? "1 of \(files.count) files can't run" : "\(count) of \(files.count) files can't run"
    }

    /// The confirmation's body: each rejected file and why, up to `limit` of them.
    package func confirmationMessage(limit: Int = 5) -> String {
        let rejected = rejected
        var lines = rejected.prefix(limit).map { "\($0.fileName): \($0.problem ?? "")" }
        if rejected.count > limit { lines.append("and \(rejected.count - limit) more") }
        return lines.joined(separator: "\n")
    }

    /// The confirming button: "Skip and run 10".
    package var skipTitle: String {
        let count = runnable.count
        return count == 1 ? "Skip and run 1 file" : "Skip and run \(count)"
    }
}

/// What submitting a batch did: the runs, in queue order, and the group their rows share.
package struct StudioBatchSubmission: Equatable {
    package let group: UUID
    package let requests: [StudioRunRequest]
    /// Files the checks passed but that could not be prepared when their turn came.
    package let failures: [StudioBatchFileCheck]
}

package enum StudioInputBatch {
    /// Why one file of a batch cannot run before any command is built: it is gone, it is a
    /// folder, the Mac will not let mere.run read it, or the slot does not take its kind.
    package static func fileProblem(
        _ path: String,
        slot: StudioAttachmentSlot,
        fileManager: FileManager = .default
    ) -> String? {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: path, isDirectory: &isDirectory) else { return "It's no longer on disk." }
        if isDirectory.boolValue { return "It's a folder, not a file." }
        guard fileManager.isReadableFile(atPath: path) else { return "mere.run isn't allowed to read it." }
        let url = URL(fileURLWithPath: path)
        guard slot.accepts(url) else {
            let kind = url.pathExtension.isEmpty ? "this kind of file" : ".\(url.pathExtension.lowercased()) files"
            return "\(slot.label) doesn't take \(kind)."
        }
        return nil
    }
}

// MARK: - Following a batch

/// Where one batch stands, read from its Library rows.
package struct StudioBatchProgress: Identifiable, Equatable {
    package let group: UUID
    /// The task the batch was run from, when known.
    package let task: StudioTask?
    package let total: Int
    package let completed: Int
    /// Failed, cancelled, or interrupted.
    package let failed: Int
    /// Running or waiting in the queue.
    package let active: Int
    /// When its first file was submitted.
    package let startedAt: Date

    package init(group: UUID, task: StudioTask?, total: Int, completed: Int, failed: Int, active: Int, startedAt: Date) {
        self.group = group
        self.task = task
        self.total = total
        self.completed = completed
        self.failed = failed
        self.active = active
        self.startedAt = startedAt
    }

    package var id: UUID { group }
    package var isActive: Bool { active > 0 }
    package var finished: Int { completed + failed }
    package var fractionFinished: Double { total == 0 ? 0 : Double(finished) / Double(total) }

    /// "4 of 12 done", with "· 1 didn't finish" when some ended without a result.
    package var summary: String {
        let done = "\(completed) of \(total) done"
        return failed == 0 ? done : "\(done) · \(failed) didn't finish"
    }

    /// "Transcribe · 12 files".
    package var title: String {
        "\(task?.title ?? "Batch") · \(total) files"
    }

    /// Every batch the rows hold, newest first. `owner` names the task a row was run from.
    package static func all(in items: [StudioLibraryItem], owner: (StudioLibraryItem) -> StudioTask?) -> [StudioBatchProgress] {
        var order: [UUID] = []
        var members: [UUID: [StudioLibraryItem]] = [:]
        for item in items {
            guard let group = item.batchGroup else { continue }
            if members[group] == nil { order.append(group) }
            members[group, default: []].append(item)
        }
        return order.compactMap { group in
            guard let rows = members[group], let first = rows.min(by: { $0.createdAt < $1.createdAt }) else { return nil }
            return StudioBatchProgress(
                group: group,
                task: owner(first),
                total: rows.count,
                completed: rows.filter { $0.status == .completed }.count,
                failed: rows.filter { [.failed, .cancelled, .interrupted].contains($0.status) }.count,
                active: rows.filter { $0.status == .running || $0.status == .queued }.count,
                startedAt: first.createdAt
            )
        }
        .sorted { $0.startedAt > $1.startedAt }
    }
}
