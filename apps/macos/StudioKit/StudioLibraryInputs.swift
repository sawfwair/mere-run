import Foundation
import UniformTypeIdentifiers

// Library runs as inputs to other tasks: which finished runs made a file an attachment entry
// point takes, so "From Library…" offers exactly what "From Disk…" would accept. The filter reads
// the entry point's own accepted types (a well slot's, or a contract path row's), never a per-page
// table, and a pick fills the slot through the same `attach` a disk pick uses.

/// What an attachment entry point takes: the slot's label, its accepted types, and whether it
/// holds a list. A well slot has one; a contract path row that binds a string builds its own.
package struct StudioAttachmentRequirement: Equatable {
    /// The slot's caption ("Audio", "Reference image"), for the picker's title and VoiceOver.
    package let label: String
    package let acceptedTypes: [UTType]
    package let allowsMultiple: Bool

    package init(label: String, acceptedTypes: [UTType], allowsMultiple: Bool = false) {
        self.label = label
        self.acceptedTypes = acceptedTypes
        self.allowsMultiple = allowsMultiple
    }

    package init(slot: StudioAttachmentSlot) {
        self.init(label: slot.label, acceptedTypes: slot.acceptedTypes, allowsMultiple: slot.allowsMultiple)
    }

    /// Whether the entry point takes a folder rather than files.
    package var picksDirectory: Bool { acceptedTypes.contains(.folder) }

    package func accepts(_ url: URL) -> Bool {
        StudioAttachmentSlot.accepts(url, acceptedTypes: acceptedTypes)
    }

    /// What the slot takes, in the words the picker's header and empty state use ("images",
    /// "videos", "audio"): the first media family any accepted type belongs to.
    package var mediaNoun: String {
        if picksDirectory { return "folders" }
        if acceptedTypes.contains(where: { $0.conforms(to: .image) }) { return "images" }
        if acceptedTypes.contains(where: { $0.conforms(to: .movie) || $0.conforms(to: .video) }) { return "videos" }
        if acceptedTypes.contains(where: { $0.conforms(to: .audio) }) { return "audio" }
        if acceptedTypes.contains(where: { $0.conforms(to: .json) }) { return "JSON files" }
        if acceptedTypes.contains(where: { $0.conforms(to: .text) }) { return "text files" }
        return "files"
    }
}

/// One finished Library run offered as an input: the run, and every file it made that the entry
/// point takes, primary output first. A run with one such file is picked whole; a run with
/// several (stems, a batch of images) lets the user pick the file.
package struct StudioLibraryInputGroup: Identifiable, Equatable {
    package let item: StudioLibraryItem
    package let files: [URL]

    package var id: UUID { item.id }
}

/// One file the picker can choose, identified by its run and path so the keyboard selection
/// survives a search that re-filters the list.
package struct StudioLibraryInputChoice: Hashable, Identifiable {
    package let itemID: UUID
    package let url: URL

    package init(itemID: UUID, url: URL) {
        self.itemID = itemID
        self.url = url
    }

    package var id: String { "\(itemID.uuidString)|\(url.path)" }
}

package enum StudioLibraryInputs {
    /// The finished runs whose files `requirement` takes, newest first, each with just those
    /// files. Conversations, runs that did not finish, and files no longer on disk are left out.
    /// `query` narrows the runs the way the Library column's search does (title, kind, prompt,
    /// model, status words), and also matches a file's name.
    package static func groups(
        in items: [StudioLibraryItem],
        for requirement: StudioAttachmentRequirement,
        query: String = "",
        titles: StudioModelTitles = .none,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) -> [StudioLibraryInputGroup] {
        let candidates: [StudioLibraryInputGroup] = items.compactMap { item in
            guard item.status == .completed, !item.isConversation else { return nil }
            let files = item.allArtifactURLs.filter { requirement.accepts($0) && fileExists($0) }
            return files.isEmpty ? nil : StudioLibraryInputGroup(item: item, files: files)
        }
        .sorted { $0.item.createdAt > $1.item.createdAt }

        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return candidates }
        let matchingRuns = Set(StudioLibraryPresenter.filter(
            candidates.map(\.item),
            with: StudioLibraryFilter(scope: .all, query: needle),
            titles: titles
        ).map(\.id))
        let lowered = needle.lowercased()
        return candidates.filter { group in
            matchingRuns.contains(group.id)
                || group.files.contains { $0.lastPathComponent.lowercased().contains(lowered) }
        }
    }

    /// Whether any finished run made a file `requirement` takes: when none did, the entry point
    /// goes straight to the open panel rather than offering an empty Library.
    package static func hasCandidates(
        in items: [StudioLibraryItem],
        for requirement: StudioAttachmentRequirement,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) -> Bool {
        items.contains { item in
            item.status == .completed && !item.isConversation
                && item.allArtifactURLs.contains { requirement.accepts($0) && fileExists($0) }
        }
    }

    /// The picker's choices in list order: one per file, a run's files in the order it lists them.
    package static func choices(in groups: [StudioLibraryInputGroup]) -> [StudioLibraryInputChoice] {
        groups.flatMap { group in group.files.map { StudioLibraryInputChoice(itemID: group.id, url: $0) } }
    }
}
