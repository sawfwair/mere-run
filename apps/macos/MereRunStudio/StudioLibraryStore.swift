import Combine
import Foundation

@MainActor
final class StudioLibraryStore: ObservableObject {
    @Published private(set) var items: [StudioLibraryItem] = []
    /// Last persistence failure, surfaced non-blockingly so silent history loss is detectable.
    @Published private(set) var lastPersistenceError: String?

    let libraryURL: URL
    private let fileManager: FileManager
    /// How a deleted row's files reach the Trash. Injected so tests can delete without a Trash.
    private let trashItem: (URL) throws -> Void

    init(
        libraryURL: URL = StudioLibraryStore.defaultLibraryURL(),
        fileManager: FileManager = .default,
        trashItem: @escaping (URL) throws -> Void = { try FileManager.default.trashItem(at: $0, resultingItemURL: nil) }
    ) {
        self.libraryURL = libraryURL
        self.fileManager = fileManager
        self.trashItem = trashItem
        load()
    }

    static func defaultLibraryURL() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("App Library", isDirectory: true)
            .appendingPathComponent("library.json", isDirectory: false)
    }

    func load() {
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
        } catch {
            items = []
            recoverCorruptLibrary()
        }
    }

    @discardableResult
    func start(
        request: StudioRunRequest,
        commandPreview: String,
        status: StudioLibraryStatus = .running
    ) -> StudioLibraryItem {
        let item = StudioLibraryItem(
            id: request.id,
            mode: request.mode,
            prompt: request.draft.prompt,
            inputURL: request.draft.inputPath.isBlank ? nil : URL(fileURLWithPath: request.draft.inputPath),
            outputURL: nil,
            createdAt: request.createdAt,
            updatedAt: Date(),
            status: status,
            exitCode: nil,
            commandPreview: commandPreview,
            outputText: nil,
            templateID: request.templateID,
            commandDraft: request.draft
        )
        upsert(item)
        return item
    }

    func markRunning(id: UUID) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.status = .running
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    func updateOutput(id: UUID, outputURL: URL) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.outputURL = outputURL
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    func updateArtifacts(id: UUID, artifactURLs: [URL]) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.artifactURLs = artifactURLs.isEmpty ? nil : artifactURLs
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    func complete(
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
    func appendUser(
        conversationID: UUID,
        mode: StudioMode,
        model: String?,
        systemPrompt: String?,
        content: String,
        imagePath: String? = nil
    ) -> StudioLibraryItem {
        if let index = items.firstIndex(where: { $0.id == conversationID }) {
            var item = items[index]
            item.messages = (item.messages ?? []) + [StudioMessage(role: .user, content: content, imagePath: imagePath)]
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
            messages: [StudioMessage(role: .user, content: content, imagePath: imagePath)],
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
    func appendAssistant(
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
            tokensPerSecond: tokensPerSecond
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
    func truncate(conversationID: UUID, removingFrom messageID: UUID) -> StudioMessage? {
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
    func branch(conversationID: UUID, at messageID: UUID, inclusive: Bool) -> StudioLibraryItem? {
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
                tokensPerSecond: message.tokensPerSecond
            )
        }
        let now = Date()
        let branch = StudioLibraryItem(
            id: UUID(),
            mode: source.mode,
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
            systemPrompt: source.systemPrompt,
            model: source.model
        )
        items.insert(branch, at: 0)
        save()
        return branch
    }

    /// Drops the last assistant message of a thread (used by retry before re-running the turn).
    func dropLastAssistant(conversationID: UUID) {
        guard let index = items.firstIndex(where: { $0.id == conversationID }) else { return }
        var item = items[index]
        guard var messages = item.messages, messages.last?.role == .assistant else { return }
        messages.removeLast()
        item.messages = messages
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    func delete(id: UUID) {
        delete(ids: [id], trashingFiles: false)
    }

    /// Removes rows, and — when the user asked for it — moves every file they produced to the
    /// Trash. Deleting the row is never blocked by a file that will not move (already deleted,
    /// on a volume with no Trash); those come back so the caller can say which.
    @discardableResult
    func delete(ids: Set<UUID>, trashingFiles: Bool) -> [URL] {
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

    func setFavorite(id: UUID, isFavorite: Bool) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        // Written as nil rather than false when unstarred, so a row that was never starred keeps
        // the shape older builds decode.
        item.isFavorite = isFavorite ? true : nil
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    func rename(id: UUID, title: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        item.customTitle = trimmed.isEmpty ? nil : trimmed
        item.updatedAt = Date()
        items[index] = item
        save()
    }

    func upsert(_ item: StudioLibraryItem) {
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
    func importReceipt(at receiptURL: URL) throws -> StudioLibraryItem {
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
    static var mereRunApp: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

extension JSONDecoder {
    static var mereRunApp: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
