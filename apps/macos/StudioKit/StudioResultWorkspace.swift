import Foundation

package struct StudioResultSelection: Codable, Equatable {
    package let itemID: UUID
    package let url: URL

    package init(itemID: UUID, url: URL) {
        self.itemID = itemID
        self.url = url
    }
}

package struct StudioResultDifference: Identifiable, Equatable {
    package let id: String
    package let title: String
    package let first: String
    package let second: String
}

package enum StudioResultComparison {
    package static func hasSettings(_ item: StudioLibraryItem) -> Bool {
        item.templateID?.capability != nil && (item.commandArguments != nil || item.commandDraft != nil)
    }

    package static func differences(_ first: StudioLibraryItem, _ second: StudioLibraryItem) -> [StudioResultDifference] {
        func form(_ item: StudioLibraryItem) -> StudioConsoleDraft? {
            guard let id = item.templateID, let template = CommandCatalog.template(id: id) else { return nil }
            return StudioExecution(templateID: id, arguments: item.commandArguments
                ?? item.commandDraft.map(template.arguments(from:)) ?? []).form
        }
        guard let a = form(first), let b = form(second) else { return [] }
        let omitted: Set<String> = ["--output", "--json-output", "--mask-output-dir", "--receipt",
                                   "--progress-json", "--api-key", "--admin-password", "--infinity-api-key",
                                   "--structured-prompt-output", "--lrc-output", "--recipe-output", "--daw-bundle", "--timings-output"]
        let labels = Dictionary((first.templateID?.capability?.options ?? []).map { ($0.flag, $0.label) },
                                uniquingKeysWith: { first, _ in first })
        return Set(a.values.keys).union(b.values.keys).subtracting(omitted).sorted().compactMap { flag in
            guard a[flag] != b[flag] else { return nil }
            return StudioResultDifference(id: flag, title: labels[flag] ?? flag,
                                          first: a.text(flag).isEmpty ? "Default" : a.text(flag),
                                          second: b.text(flag).isEmpty ? "Default" : b.text(flag))
        }
    }
}

package enum StudioResultContinuation: String, CaseIterable, Identifiable {
    case edit, reference, animate, read, segment
    package var id: String { rawValue }
    package var title: String {
        switch self {
        case .edit: "Edit image"
        case .reference: "Use as reference"
        case .animate: "Make a video"
        case .read: "Understand image"
        case .segment: "Select a subject"
        }
    }
    package var symbol: String {
        switch self {
        case .edit: "slider.horizontal.3"
        case .reference: "photo.on.rectangle"
        case .animate: "film"
        case .read: "eye"
        case .segment: "viewfinder"
        }
    }
    package var task: StudioTask {
        switch self {
        case .edit, .reference: .imageGenerate
        case .animate: .videoGenerate
        case .read: .visionRead
        case .segment: .visionSegment
        }
    }

    package func draft(from item: StudioLibraryItem, url: URL, baseline: StudioDraft) -> StudioDraft? {
        guard StudioOutputFileKind.classify(url) == .image else { return nil }
        var draft = baseline
        draft.parentID = item.id
        draft.referenceImagePaths = ""
        draft.prompt = self == .read ? "Describe this image." : item.prompt
        if self == .reference {
            draft.referenceImagePaths = url.path
            draft.inputPath = ""
        } else {
            draft.inputPath = url.path
        }
        if self == .edit {
            draft.width = item.commandDraft?.width ?? baseline.width
            draft.height = item.commandDraft?.height ?? baseline.height
        }
        return draft
    }
}

package enum StudioFileExport {
    /// Copies before replacing the destination. Saving onto the source is a successful no-op.
    package static func copy(_ source: URL, to destination: URL, fileManager: FileManager = .default) throws {
        guard source.resolvingSymlinksInPath().standardizedFileURL
                != destination.resolvingSymlinksInPath().standardizedFileURL else { return }
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".mere-export-\(UUID())")
        try fileManager.copyItem(at: source, to: temporary)
        defer { try? fileManager.removeItem(at: temporary) }
        if fileManager.fileExists(atPath: destination.path) {
            _ = try fileManager.replaceItemAt(destination, withItemAt: temporary)
        } else {
            try fileManager.moveItem(at: temporary, to: destination)
        }
    }
}
