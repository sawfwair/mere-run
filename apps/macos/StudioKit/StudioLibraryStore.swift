import Combine
import Foundation

@MainActor
package final class StudioLibraryStore: ObservableObject {
    @Published package private(set) var items: [StudioLibraryItem] = []
    /// Last persistence failure, surfaced non-blockingly so silent history loss is detectable.
    @Published package private(set) var lastPersistenceError: String?

    package let libraryURL: URL
    private let fileManager: FileManager
    private weak var observedController: MereRunController?
    private var subscriptions = Set<AnyCancellable>()
    private var completedRequests = Set<UUID>()
    /// How a deleted row's files reach the Trash. Injected so tests can delete without a Trash.
    private let trashItem: (URL) throws -> Void

    package init(
        libraryURL: URL = StudioLibraryStore.defaultLibraryURL(),
        fileManager: FileManager = .default,
        trashItem: @escaping (URL) throws -> Void = { try FileManager.default.trashItem(at: $0, resultingItemURL: nil) }
    ) {
        self.libraryURL = libraryURL
        self.fileManager = fileManager
        self.trashItem = trashItem
        load()
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
            if let conversationID = result.conversationID {
                let text = result.outputText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                self.appendAssistant(
                    conversationID: conversationID,
                    content: text.isEmpty ? (result.exitCode == 0 ? "(No output.)" : "Run stopped before a reply was received.") : text,
                    exitCode: result.exitCode,
                    model: job?.request.draft?.model,
                    systemPrompt: job?.request.draft?.secondaryText,
                    tokensPerSecond: job.flatMap { ConversationTranscript.decodeTokensPerSecond(in: $0.log.lines.map(\.text)) }
                )
            } else {
                self.complete(id: requestID, exitCode: result.exitCode, outputURL: result.outputURL,
                              outputText: result.outputText, commandPreview: result.commandPreview.maskingAPIKeyValue(),
                              artifactURLs: result.artifactURLs, artifactRoles: result.artifactRoles)
            }
            if let job, case .cancelled = job.state,
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
            case .output, .finished: break
            }
        }.store(in: &subscriptions)
    }

    package func load() {
        do {
            guard fileManager.fileExists(atPath: libraryURL.path) else {
                items = []
                return
            }

            let data = try Data(contentsOf: libraryURL)
            // Decode leniently per row: one un-decodable entry (e.g. written by a newer/older
            // build) must not discard the entire history. Only a top-level parse failure (not an
            // array at all) falls through to corrupt-file recovery.
            let rows = try JSONDecoder.mereRunApp.decode([FailableDecodable<StudioLibraryItem>].self, from: data)
            items = rows.compactMap(\.value).sorted { $0.createdAt > $1.createdAt }
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
            recoverCorruptLibrary()
        }
    }

    @discardableResult
    package func start(
        request: StudioRunRequest,
        commandPreview: String,
        status: StudioLibraryStatus = .running,
        arguments: [String]? = nil
    ) -> StudioLibraryItem {
        let execution = request.execution ?? StudioExecution(
            templateID: request.templateID,
            arguments: arguments ?? request.template.arguments(from: request.draft)
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
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.status = .running
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
        if let outputURL {
            item.outputURL = outputURL
        }
        if !artifactURLs.isEmpty {
            item.artifactURLs = artifactURLs
        }
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

    /// Appends the assistant reply for the latest turn, recording the model and system prompt
    /// that produced it (and the decode speed when the run reported one). A non-zero exit marks
    /// the message failed but keeps the thread so the user can retry.
    package func appendAssistant(
        conversationID: UUID,
        content: String,
        exitCode: Int32,
        model: String? = nil,
        systemPrompt: String? = nil,
        tokensPerSecond: Double? = nil
    ) {
        guard let index = items.firstIndex(where: { $0.id == conversationID }) else { return }
        var item = items[index]
        var messages = item.messages ?? []
        messages.append(StudioMessage(
            role: .assistant,
            content: content,
            failed: exitCode != 0,
            model: model,
            systemPrompt: systemPrompt,
            tokensPerSecond: tokensPerSecond,
            preset: item.mode
        ))
        item.messages = messages
        item.status = exitCode == 0 ? .completed : .failed
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
                imagePath: message.imagePath,
                model: message.model,
                systemPrompt: message.systemPrompt,
                tokensPerSecond: message.tokensPerSecond,
                preset: message.preset
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
            status: kept.last?.failed == true ? .failed : .completed,
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

    package func delete(id: UUID) {
        delete(ids: [id], trashingFiles: false)
    }

    /// Removes rows, and — when the user asked for it — moves every file they produced to the
    /// Trash. Deleting the row is never blocked by a file that will not move (already deleted,
    /// on a volume with no Trash); those come back so the caller can say which.
    @discardableResult
    package func delete(ids: Set<UUID>, trashingFiles: Bool) -> [URL] {
        guard !ids.isEmpty else { return [] }
        var failures: [URL] = []
        if trashingFiles {
            var seen = Set<URL>()
            for item in items where ids.contains(item.id) {
                for url in item.allArtifactURLs where seen.insert(url.standardizedFileURL).inserted {
                    guard fileManager.fileExists(atPath: url.path) else { continue }
                    do {
                        try trashItem(url)
                    } catch {
                        failures.append(url)
                    }
                }
            }
        }
        items.removeAll { ids.contains($0.id) }
        save()
        return failures
    }

    package func setFavorite(id: UUID, isFavorite: Bool) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        // Written as nil rather than false when unstarred, so a row that was never starred keeps
        // the shape older builds decode.
        item.isFavorite = isFavorite ? true : nil
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    package func rename(id: UUID, title: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        item.customTitle = trimmed.isEmpty ? nil : trimmed
        item.updatedAt = Date()
        items[index] = item
        save()
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
            let data = try JSONEncoder.mereRunApp.encode(items)
            try data.write(to: libraryURL, options: [.atomic])
            lastPersistenceError = nil
        } catch {
            // Persistence must never block local generation, but the failure is surfaced
            // so the UI can warn that run history may not survive relaunch.
            lastPersistenceError = error.localizedDescription
        }
    }

    private func recoverCorruptLibrary() {
        guard fileManager.fileExists(atPath: libraryURL.path) else { return }
        let recoveryURL = libraryURL
            .deletingPathExtension()
            .appendingPathExtension("corrupt-\(DateFormatter.mereRunTimestamp.string(from: Date()))")
            .appendingPathExtension("json")
        try? fileManager.moveItem(at: libraryURL, to: recoveryURL)
    }
}

/// Decodes `T` if possible, otherwise resolves to nil instead of throwing — lets an array decode
/// skip individual bad elements without losing the rest.
private struct FailableDecodable<T: Decodable>: Decodable {
    let value: T?

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        value = try? container.decode(T.self)
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
