import Combine
import Foundation

@MainActor
package final class StudioLibraryStore: ObservableObject {
    @Published package private(set) var items: [StudioLibraryItem] = []
    /// Last persistence failure, surfaced non-blockingly so silent history loss is detectable.
    @Published package private(set) var lastPersistenceError: String?
    /// Rows in the library file this build cannot read: written by a different version, or
    /// holding a field of an unexpected type. They ride through every save untouched and are
    /// counted here so the UI can say history exists that is not shown.
    @Published package private(set) var preservedRowCount = 0

    package let libraryURL: URL
    private let fileManager: FileManager
    private weak var observedController: MereRunController?
    private var subscriptions = Set<AnyCancellable>()
    private var completedRequests = Set<UUID>()
    private var preservedRows: [StudioLibraryJSON] = []
    /// The file is copied aside once per launch before the first rewrite that carries
    /// preserved rows, so an untouched original always exists.
    private var hasQuarantinedOriginal = false
    /// How a deleted row's files reach the Trash, answering where each landed so Undo can move
    /// it back. Injected so tests can delete without touching the real Trash.
    private let trashItem: (URL) throws -> URL?
    /// Where deletions, renames, and favorites register their undo steps.
    package let undo = StudioUndo()

    package init(
        libraryURL: URL = StudioLibraryStore.defaultLibraryURL(),
        fileManager: FileManager = .default,
        trashItem: @escaping (URL) throws -> URL? = StudioLibraryStore.moveToTrash
    ) {
        self.libraryURL = libraryURL
        self.fileManager = fileManager
        self.trashItem = trashItem
        load()
    }

    /// What the UI shows when the file holds rows this build cannot read.
    package var preservationNotice: String? {
        switch preservedRowCount {
        case 0:
            return nil
        case 1:
            return "1 history entry can't be read by this version of mere.run. It's kept in the file but not shown."
        default:
            return "\(preservedRowCount) history entries can't be read by this version of mere.run. They're kept in the file but not shown."
        }
    }

    nonisolated package static func moveToTrash(_ url: URL) throws -> URL? {
        var resulting: NSURL?
        try FileManager.default.trashItem(at: url, resultingItemURL: &resulting)
        return resulting as URL?
    }

    package static func defaultLibraryURL() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("App Library", isDirectory: true)
            .appendingPathComponent("library.json", isDirectory: false)
    }

    /// Keeps durable history current even with every Studio window closed.
    package func observe(controller: MereRunController) {
        guard observedController !== controller else { return }
        subscriptions.removeAll()
        completedRequests.removeAll()
        observedController = controller
        controller.runCompletions.sink { [weak self, weak controller] result in
            guard let self, let controller, let requestID = result.requestID,
                  self.completedRequests.insert(requestID).inserted else { return }
            let job = controller.jobs.job(requestID: requestID)
            let cancelled: Bool
            if case .cancelled = job?.state { cancelled = true } else { cancelled = false }
            if let conversationID = result.conversationID {
                let text = result.outputText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let failed = result.exitCode != 0 && !cancelled
                // A run that never started has no reply and no log: its preflight message is
                // the reason, once.
                var preflight: JobPreflightFailure?
                if case .preflightFailed(let failure) = job?.state { preflight = failure }
                self.appendAssistant(
                    conversationID: conversationID,
                    content: text.isEmpty || preflight != nil
                        ? Self.placeholderReply(exitCode: result.exitCode, cancelled: cancelled) : text,
                    exitCode: result.exitCode,
                    cancelled: cancelled,
                    model: job?.request.draft?.model,
                    systemPrompt: job?.request.draft?.secondaryText,
                    tokensPerSecond: job.flatMap { ConversationTranscript.decodeTokensPerSecond(in: $0.log.lines.map(\.text)) },
                    reasoning: result.reasoning,
                    failureReason: failed ? preflight?.message ?? Self.failureReason(job: job, exitCode: result.exitCode) : nil,
                    logTail: failed && preflight == nil ? job.map { Self.logTail(of: $0, exitCode: result.exitCode) } : nil
                )
            } else {
                self.complete(id: requestID, exitCode: result.exitCode, outputURL: result.outputURL,
                              outputText: result.outputText, commandPreview: result.commandPreview.maskingAPIKeyValue(),
                              artifactURLs: result.artifactURLs, artifactRoles: result.artifactRoles)
            }
            if cancelled,
               let index = self.items.firstIndex(where: { $0.id == (result.conversationID ?? requestID) }) {
                self.items[index].status = .cancelled
                self.save()
            }
        }.store(in: &subscriptions)
        controller.jobs.events.sink { [weak self] event in
            guard let self else { return }
            switch event {
            case .started(let job):
                guard let id = job.request.requestID else { return }
                self.markRunning(id: job.request.conversationID ?? id)
            case .changed(let job):
                guard let id = job.request.requestID, let output = job.primaryArtifactURL,
                      self.items.first(where: { $0.id == id })?.outputURL != output else { return }
                self.updateOutput(id: id, outputURL: output)
            case .queued, .reordered, .output, .finished: break
            }
        }.store(in: &subscriptions)
    }

    package func load() {
        do {
            guard fileManager.fileExists(atPath: libraryURL.path) else {
                items = []
                setPreservedRows([])
                return
            }

            let data = try Data(contentsOf: libraryURL)
            // Decode per row: an entry this build cannot read (written by a newer or older build)
            // is kept as JSON rather than discarded, and never discards the rest of the history.
            // Only a top-level parse failure (not an array at all) falls through to corrupt-file
            // recovery.
            let rows = try JSONDecoder.mereRunApp.decode([StudioLibraryRow].self, from: data)
            var loaded: [StudioLibraryItem] = []
            var preserved: [StudioLibraryJSON] = []
            for row in rows {
                switch row {
                case .item(let item): loaded.append(item)
                case .preserved(let json): preserved.append(json)
                }
            }
            items = loaded.sorted { $0.createdAt > $1.createdAt }
            setPreservedRows(preserved)
            var reconciled = false
            for index in items.indices where items[index].status == .running || items[index].status == .queued {
                let item = items[index]
                let owned = observedController?.jobs.all.contains {
                    $0.state.isActive && ($0.request.requestID == item.id || $0.request.conversationID == item.id)
                } ?? false
                if !owned {
                    items[index].status = .interrupted
                    reconciled = true
                }
            }
            if reconciled { save() }
        } catch {
            items = []
            setPreservedRows([])
            recoverCorruptLibrary()
        }
    }

    private func setPreservedRows(_ rows: [StudioLibraryJSON]) {
        preservedRows = rows
        preservedRowCount = rows.count
    }

    @discardableResult
    package func start(
        request: StudioRunRequest,
        commandPreview: String,
        status: StudioLibraryStatus = .running,
        arguments: [String]? = nil,
        source: StudioScopeSource
    ) -> StudioLibraryItem {
        let execution = request.execution ?? StudioExecution(
            templateID: request.templateID,
            arguments: arguments ?? request.template.arguments(from: request.draft, source: source)
        )
        let recordedDraft = (request.execution == nil && arguments == nil
            ? request.draft : execution.project(onto: request.draft)).withoutSecrets
        var item = StudioLibraryItem(
            id: request.id,
            mode: request.mode,
            prompt: recordedDraft.prompt,
            inputURL: recordedDraft.inputPath.isBlank ? nil : URL(fileURLWithPath: recordedDraft.inputPath),
            outputURL: nil,
            createdAt: request.createdAt,
            updatedAt: Date(),
            status: status,
            exitCode: nil,
            commandPreview: commandPreview,
            outputText: nil,
            templateID: request.templateID,
            commandDraft: recordedDraft,
            commandArguments: execution.arguments.maskingSecrets(),
            parentID: request.parentID
        )
        item.inputIdentity = item.inputURL.flatMap(StudioInputIdentity.read)
        upsert(item)
        return item
    }

    package func markRunning(id: UUID) {
        setStatus(.running, id: id)
    }

    package func setStatus(_ status: StudioLibraryStatus, id: UUID) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.status = status
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    package func updateOutput(id: UUID, outputURL: URL) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.outputURL = outputURL
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    package func updateArtifacts(id: UUID, artifactURLs: [URL]) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.artifactURLs = artifactURLs.isEmpty ? nil : artifactURLs
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    package func complete(
        id: UUID,
        exitCode: Int32,
        outputURL: URL?,
        outputText: String?,
        commandPreview: String,
        artifactURLs: [URL] = [],
        artifactRoles: [String: String] = [:]
    ) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.status = exitCode == 0 ? .completed : .failed
        item.exitCode = exitCode
        item.updatedAt = Date()
        item.commandPreview = commandPreview
        item.outputURL = outputURL
        item.artifactURLs = artifactURLs.isEmpty ? nil : artifactURLs
        item.artifactRoles = artifactRoles.isEmpty ? nil : artifactRoles
        item.outputText = outputText

        if shouldKeep(item) {
            items[index] = item
        } else {
            items.remove(at: index)
        }
        save()
    }

    /// Appends a user turn to a chat/code conversation, creating the thread item lazily on the
    /// first message (so a "New chat" that is never sent leaves no empty row). The thread-level
    /// preset, model, and system prompt follow the latest turn, so retries replay the settings
    /// the user last chose; which turn ran with what is recorded on the assistant turns.
    @discardableResult
    package func appendUser(
        conversationID: UUID,
        mode: StudioMode,
        model: String?,
        systemPrompt: String?,
        content: String,
        imagePath: String? = nil
    ) -> StudioLibraryItem {
        if let index = items.firstIndex(where: { $0.id == conversationID }) {
            var item = items[index]
            item.messages = (item.messages ?? []) + [StudioMessage(role: .user, content: content, imagePath: imagePath, model: model, systemPrompt: systemPrompt, preset: mode)]
            item.mode = mode
            item.model = model
            item.systemPrompt = systemPrompt
            item.commandPreview = mode == .code ? "mere.run text code" : "mere.run text chat"
            item.status = .running
            item.updatedAt = Date()
            items[index] = item
            save()
            return item
        }

        let item = StudioLibraryItem(
            id: conversationID,
            mode: mode,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: Date(),
            updatedAt: Date(),
            status: .running,
            exitCode: nil,
            commandPreview: mode == .code ? "mere.run text code" : "mere.run text chat",
            outputText: nil,
            messages: [StudioMessage(role: .user, content: content, imagePath: imagePath, model: model, systemPrompt: systemPrompt, preset: mode)],
            systemPrompt: systemPrompt,
            model: model
        )
        items.insert(item, at: 0)
        save()
        return item
    }

    /// How many lines of a failed turn's stderr ride along in the thread.
    package static let turnLogTailLines = 40

    /// What an assistant turn says when the run produced no reply text. A failed turn says
    /// nothing here — its `failureReason` line says why.
    private static func placeholderReply(exitCode: Int32, cancelled: Bool) -> String {
        if exitCode == 0 { return "(No output.)" }
        return cancelled ? "Run stopped before a reply was received." : ""
    }

    /// One line saying why a conversation turn failed: the last meaningful line of its stderr,
    /// else the exit code.
    private static func failureReason(job: Job?, exitCode: Int32) -> String {
        StudioFailureSummary.summary(outputText: nil, logLines: job?.log.text(of: [.stderr]) ?? [], exitCode: exitCode)
    }

    /// The last lines a failed run wrote to stderr, secrets masked, closed by its exit note. The
    /// launched command line stays out: for a turn it carries the whole rendered thread.
    private static func logTail(of job: Job, exitCode: Int32) -> [String] {
        let lines = job.log.text(of: [.stderr]).suffix(turnLogTailLines).map { $0.maskingSecretValues() }
        return lines + ["Exited with code \(exitCode)."]
    }

    /// Appends the assistant reply for the latest turn, recording the model and system prompt
    /// that produced it (and the decode speed when the run reported one). A non-zero exit marks
    /// the message failed but keeps the thread so the user can retry; the failure's reason and
    /// log tail, like the turn's reasoning, are kept for display and never replayed.
    package func appendAssistant(
        conversationID: UUID,
        content: String,
        exitCode: Int32,
        cancelled: Bool = false,
        model: String? = nil,
        systemPrompt: String? = nil,
        tokensPerSecond: Double? = nil,
        reasoning: String? = nil,
        failureReason: String? = nil,
        logTail: [String]? = nil
    ) {
        guard let index = items.firstIndex(where: { $0.id == conversationID }) else { return }
        var item = items[index]
        var messages = item.messages ?? []
        messages.append(StudioMessage(
            role: .assistant,
            content: content,
            failed: exitCode != 0,
            cancelled: cancelled ? true : nil,
            model: model,
            systemPrompt: systemPrompt,
            tokensPerSecond: tokensPerSecond,
            preset: item.mode,
            reasoning: reasoning,
            failureReason: failureReason,
            logTail: logTail
        ))
        item.messages = messages
        item.status = cancelled ? .cancelled : (exitCode == 0 ? .completed : .failed)
        item.exitCode = exitCode
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    /// Truncates a thread at `messageID` (removing it and everything after). Returns the removed
    /// message when it was a user turn, so the composer can be repopulated (text + image) for editing.
    @discardableResult
    package func truncate(conversationID: UUID, removingFrom messageID: UUID) -> StudioMessage? {
        guard let index = items.firstIndex(where: { $0.id == conversationID }),
              var messages = items[index].messages,
              let messageIndex = messages.firstIndex(where: { $0.id == messageID }) else { return nil }
        let removed = messages[messageIndex]
        messages.removeSubrange(messageIndex...)
        var item = items[index]
        item.messages = messages
        item.updatedAt = Date()
        items[index] = item
        save()
        return removed.role == .user ? removed : nil
    }

    /// Starts a new thread from a point in an existing one: the messages before `messageID`,
    /// plus that message itself when `inclusive`. Branching from a user turn (exclusive) leaves
    /// the original untouched and gives the edited turn a fresh thread; branching from an
    /// assistant turn (inclusive) forks the conversation after that reply. The branch inherits
    /// the source's preset, model, and system prompt and gets its own message identities.
    @discardableResult
    package func branch(conversationID: UUID, at messageID: UUID, inclusive: Bool) -> StudioLibraryItem? {
        guard let source = items.first(where: { $0.id == conversationID }),
              let messages = source.messages,
              let messageIndex = messages.firstIndex(where: { $0.id == messageID }) else { return nil }
        let end = inclusive ? messageIndex + 1 : messageIndex
        let kept = messages[..<end].map { message in
            StudioMessage(
                role: message.role,
                content: message.content,
                createdAt: message.createdAt,
                failed: message.failed,
                cancelled: message.cancelled,
                imagePath: message.imagePath,
                model: message.model,
                systemPrompt: message.systemPrompt,
                tokensPerSecond: message.tokensPerSecond,
                preset: message.preset,
                reasoning: message.reasoning,
                failureReason: message.failureReason,
                logTail: message.logTail
            )
        }
        // An edited user turn adopts that turn's settings, even though it is excluded from history.
        let point = messages[messageIndex]
        let effective = point.preset != nil ? point : messages[...messageIndex].last { $0.model != nil || $0.systemPrompt != nil }
        let now = Date()
        let branch = StudioLibraryItem(
            id: UUID(),
            mode: effective?.preset ?? source.mode,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: now,
            updatedAt: now,
            status: kept.last?.cancelled == true ? .cancelled : (kept.last?.failed == true ? .failed : .completed),
            exitCode: kept.last?.failed == true ? 1 : 0,
            commandPreview: source.commandPreview,
            outputText: nil,
            messages: kept,
            systemPrompt: effective.map(\.systemPrompt) ?? source.systemPrompt,
            model: effective.map(\.model) ?? source.model
        )
        items.insert(branch, at: 0)
        save()
        return branch
    }

    /// Drops the last assistant message of a thread (used by retry before re-running the turn).
    package func dropLastAssistant(conversationID: UUID) {
        guard let index = items.firstIndex(where: { $0.id == conversationID }) else { return }
        var item = items[index]
        guard var messages = item.messages, messages.last?.role == .assistant else { return }
        messages.removeLast()
        item.messages = messages
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    /// Removes a row the app itself emptied (a thread whose only turn was taken back for
    /// editing). Not an undo step: the user did not ask for a deletion.
    package func delete(id: UUID) {
        _ = remove(ids: [id], trashingFiles: false)
    }

    /// Library ▸ Delete: removes rows, and — when the user asked for it — moves every file they
    /// produced to the Trash. Deleting the row is never blocked by a file that will not move
    /// (already deleted, on a volume with no Trash); those come back so the caller can say which.
    ///
    /// The deletion is written at once and is one undo step. Undo puts the rows back where they
    /// were and moves the files back out of the Trash; the files never wait anywhere the app
    /// would have to clean up, so a deletion nobody undoes is exactly what the user asked for.
    @discardableResult
    package func delete(ids: Set<UUID>, trashingFiles: Bool) -> [URL] {
        let removal = remove(ids: ids, trashingFiles: trashingFiles)
        guard !removal.rows.isEmpty else { return removal.failures }
        let name = Self.deletionUndoName(removal.rows.map(\.item))
        undo.register(name) { [weak self] in self?.restore(removal, name: name, trashingFiles: trashingFiles) }
        return removal.failures
    }

    /// The Edit menu's name for deleting `items`: "Delete Run", "Delete Threads", "Delete Items".
    package static func deletionUndoName(_ items: [StudioLibraryItem]) -> String {
        let noun: String
        if items.allSatisfy(\.isConversation) {
            noun = "Thread"
        } else if items.allSatisfy({ !$0.isConversation }) {
            noun = "Run"
        } else {
            noun = "Item"
        }
        return "Delete " + noun + (items.count == 1 ? "" : "s")
    }

    private struct Removal {
        struct Row {
            let index: Int
            let item: StudioLibraryItem
        }

        struct TrashedFile {
            let original: URL
            let trashed: URL
        }

        let rows: [Row]
        let trashed: [TrashedFile]
        let failures: [URL]
    }

    private func remove(ids: Set<UUID>, trashingFiles: Bool) -> Removal {
        let rows = items.enumerated().filter { ids.contains($0.element.id) }.map { Removal.Row(index: $0.offset, item: $0.element) }
        guard !rows.isEmpty else { return Removal(rows: [], trashed: [], failures: []) }
        var trashed: [Removal.TrashedFile] = []
        var failures: [URL] = []
        if trashingFiles {
            var seen = Set<URL>()
            for row in rows {
                for url in row.item.allArtifactURLs where seen.insert(url.standardizedFileURL).inserted {
                    guard fileManager.fileExists(atPath: url.path) else { continue }
                    do {
                        if let landed = try trashItem(url) { trashed.append(.init(original: url, trashed: landed)) }
                    } catch {
                        failures.append(url)
                    }
                }
            }
        }
        items.removeAll { ids.contains($0.id) }
        save()
        return Removal(rows: rows, trashed: trashed, failures: failures)
    }

    /// Undo of a deletion: every file still in the Trash goes back where it was, and every row
    /// returns to its place. A file emptied from the Trash since stays gone; its row comes back
    /// all the same, the way a row whose file was deleted in Finder reads.
    private func restore(_ removal: Removal, name: String, trashingFiles: Bool) {
        for file in removal.trashed where !fileManager.fileExists(atPath: file.original.path) {
            try? fileManager.createDirectory(at: file.original.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fileManager.moveItem(at: file.trashed, to: file.original)
        }
        for row in removal.rows.sorted(by: { $0.index < $1.index }) where !items.contains(where: { $0.id == row.item.id }) {
            items.insert(row.item, at: min(row.index, items.count))
        }
        save()
        let ids = Set(removal.rows.map(\.item.id))
        undo.register(name) { [weak self] in self?.delete(ids: ids, trashingFiles: trashingFiles) }
    }

    package func setFavorite(id: UUID, isFavorite: Bool) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        let wasFavorite = item.isFavorite == true
        // Written as nil rather than false when unstarred, so a row that was never starred keeps
        // the shape older builds decode.
        item.isFavorite = isFavorite ? true : nil
        item.updatedAt = Date()
        items[index] = item
        save()
        guard wasFavorite != isFavorite else { return }
        undo.register(isFavorite ? "Add to Favorites" : "Remove from Favorites") { [weak self] in
            self?.setFavorite(id: id, isFavorite: wasFavorite)
        }
    }

    package func rename(id: UUID, title: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        let previous = item.customTitle
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        item.customTitle = trimmed.isEmpty ? nil : trimmed
        item.updatedAt = Date()
        items[index] = item
        save()
        guard item.customTitle != previous else { return }
        undo.register("Rename") { [weak self] in self?.rename(id: id, title: previous ?? "") }
    }

    package func upsert(_ item: StudioLibraryItem) {
        if let index = items.firstIndex(where: { $0.id == item.id }) {
            items[index] = item
        } else {
            items.insert(item, at: 0)
        }
        save()
    }

    /// Imports one completed launcher artifact through MereRun's typed receipt contract. The
    /// Library remains the only writer of its persisted state; launchers never edit library.json.
    @discardableResult
    package func importReceipt(at receiptURL: URL) throws -> StudioLibraryItem {
        let receipt = try StudioLibraryImportReceipt.load(from: receiptURL, fileManager: fileManager)
        let artifactURL = try receipt.artifactURL(fileManager: fileManager)

        if let existing = items.first(where: { $0.id == receipt.id }) {
            guard existing.outputURL?.standardizedFileURL == artifactURL else {
                throw StudioLibraryImportError.receiptIDConflict(receipt.id)
            }
            return existing
        }
        if let existing = items.first(where: { $0.outputURL?.standardizedFileURL == artifactURL }) {
            return existing
        }

        var item = StudioLibraryItem(
            id: receipt.id,
            mode: receipt.kind.mode,
            prompt: receipt.prompt,
            inputURL: nil,
            outputURL: artifactURL,
            createdAt: receipt.createdAt,
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: receipt.kind.commandPreview,
            outputText: nil,
            templateID: receipt.kind.mode.defaultTemplateID,
            artifactURLs: [artifactURL]
        )
        item.source = receipt.source
        upsert(item)
        return item
    }

    private func shouldKeep(_ item: StudioLibraryItem) -> Bool {
        item.status == .completed
            || item.outputURL != nil
            || item.outputText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            || !item.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || item.inputURL != nil
    }

    private func save() {
        do {
            try fileManager.createDirectory(
                at: libraryURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if !preservedRows.isEmpty, !hasQuarantinedOriginal {
                try quarantineOriginal()
                hasQuarantinedOriginal = true
            }
            let rows = items.map(StudioLibraryRow.item) + preservedRows.map(StudioLibraryRow.preserved)
            let data = try JSONEncoder.mereRunApp.encode(rows)
            try data.write(to: libraryURL, options: [.atomic])
            lastPersistenceError = nil
        } catch {
            // Persistence must never block local generation, but the failure is surfaced
            // so the UI can warn that run history may not survive relaunch.
            lastPersistenceError = error.localizedDescription
        }
    }

    /// Copies the file as this launch found it next to the library, named like a corrupt-file
    /// recovery, before the first rewrite that carries rows this build cannot read. A launch in
    /// the same second as the last backup finds the copy already there and keeps it.
    private func quarantineOriginal() throws {
        let backupURL = Self.siblingURL(of: libraryURL, tag: "preserved")
        guard !fileManager.fileExists(atPath: backupURL.path) else { return }
        try fileManager.copyItem(at: libraryURL, to: backupURL)
    }

    private func recoverCorruptLibrary() {
        guard fileManager.fileExists(atPath: libraryURL.path) else { return }
        try? fileManager.moveItem(at: libraryURL, to: Self.siblingURL(of: libraryURL, tag: "corrupt"))
    }

    private static func siblingURL(of libraryURL: URL, tag: String) -> URL {
        libraryURL
            .deletingPathExtension()
            .appendingPathExtension("\(tag)-\(DateFormatter.mereRunTimestamp.string(from: Date()))")
            .appendingPathExtension("json")
    }
}

extension JSONEncoder {
    package static var mereRunApp: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

extension JSONDecoder {
    package static var mereRunApp: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
