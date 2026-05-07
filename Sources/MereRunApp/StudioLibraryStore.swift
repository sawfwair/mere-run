import Combine
import Foundation

@MainActor
final class StudioLibraryStore: ObservableObject {
    @Published private(set) var items: [StudioLibraryItem] = []

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
    func start(request: StudioRunRequest, commandPreview: String) -> StudioLibraryItem {
        let item = StudioLibraryItem(
            id: request.id,
            mode: request.mode,
            prompt: request.draft.prompt,
            inputURL: request.draft.inputPath.isBlank ? nil : URL(fileURLWithPath: request.draft.inputPath),
            outputURL: request.expectedOutputURL,
            createdAt: request.createdAt,
            updatedAt: Date(),
            status: .running,
            exitCode: nil,
            commandPreview: commandPreview,
            outputText: nil
        )
        upsert(item)
        return item
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
        } catch {
            // Library persistence should never block local generation.
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
