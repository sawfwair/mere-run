import Combine
import Foundation

@MainActor
final class StudioLibraryStore: ObservableObject {
    @Published private(set) var items: [StudioLibraryItem] = []
    /// Last persistence failure, surfaced non-blockingly so silent history loss is detectable.
    @Published private(set) var lastPersistenceError: String?

    let libraryURL: URL
    private let fileManager: FileManager

    init(
        libraryURL: URL = StudioLibraryStore.defaultLibraryURL(),
        fileManager: FileManager = .default
    ) {
        self.libraryURL = libraryURL
        self.fileManager = fileManager
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
            items = try JSONDecoder.mereRunApp.decode([StudioLibraryItem].self, from: data)
                .sorted { $0.createdAt > $1.createdAt }
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
            outputText: nil
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

    func complete(
        id: UUID,
        exitCode: Int32,
        outputURL: URL?,
        outputText: String?,
        commandPreview: String
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
        item.outputText = outputText

        if shouldKeep(item) {
            items[index] = item
        } else {
            items.remove(at: index)
        }
        save()
    }

    /// Appends a user turn to a chat/code conversation, creating the thread item lazily on the
    /// first message (so a "New chat" that is never sent leaves no empty row). System prompt and
    /// model are captured on the item so later turns and retries replay with the same settings.
    @discardableResult
    func appendUser(
        conversationID: UUID,
        mode: StudioMode,
        model: String?,
        systemPrompt: String?,
        content: String
    ) -> StudioLibraryItem {
        if let index = items.firstIndex(where: { $0.id == conversationID }) {
            var item = items[index]
            item.messages = (item.messages ?? []) + [StudioMessage(role: .user, content: content)]
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
            messages: [StudioMessage(role: .user, content: content)],
            systemPrompt: systemPrompt,
            model: model
        )
        items.insert(item, at: 0)
        save()
        return item
    }

    /// Appends the assistant reply for the latest turn. A non-zero exit marks the message failed
    /// but keeps the thread so the user can retry.
    func appendAssistant(conversationID: UUID, content: String, exitCode: Int32) {
        guard let index = items.firstIndex(where: { $0.id == conversationID }) else { return }
        var item = items[index]
        var messages = item.messages ?? []
        messages.append(StudioMessage(role: .assistant, content: content, failed: exitCode != 0))
        item.messages = messages
        item.status = exitCode == 0 ? .completed : .failed
        item.exitCode = exitCode
        item.updatedAt = Date()
        items[index] = item
        save()
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
        items.removeAll { $0.id == id }
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
