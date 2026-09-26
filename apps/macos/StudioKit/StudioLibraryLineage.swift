import Foundation
import MereRunContract

// Provenance between Library rows: a run "made from" the rows whose output files it read, and a
// row "used in" the runs that read it. However the file reached the input (From Library, Send to,
// a drag, or a path typed by hand) the run ends up naming it, so one path match covers them all.
//
// A run records its sources when it is submitted (`StudioLibraryItem.sourceItemIDs`). Rows from
// before that have none recorded, so their links are inferred from their recorded input paths
// against every earlier row's outputs; that happens when the index is first read, never on load,
// and is never written back.

/// Which rows each row was made from and used in, built once per Library change.
package struct StudioLibraryLineage: Equatable {
    /// Made from: each row's sources, in the order its inputs name them.
    package private(set) var sources: [UUID: [UUID]] = [:]
    /// Used in: each row's consumers, newest first.
    package private(set) var uses: [UUID: [UUID]] = [:]
    private var itemsByID: [UUID: StudioLibraryItem] = [:]

    package static let empty = StudioLibraryLineage(items: [])

    package init(items: [StudioLibraryItem]) {
        let rows = items.filter { !$0.isConversation }
        itemsByID = Dictionary(rows.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        let producers = Self.producers(in: rows)
        for item in rows {
            let found = (item.sourceItemIDs ?? Self.inferredSources(for: item, producers: producers))
                .filter { $0 != item.id && itemsByID[$0] != nil }
            guard !found.isEmpty else { continue }
            sources[item.id] = found
            for source in found { uses[source, default: []].append(item.id) }
        }
        for (id, consumers) in uses {
            uses[id] = consumers.sorted { (itemsByID[$0]?.createdAt ?? .distantPast) > (itemsByID[$1]?.createdAt ?? .distantPast) }
        }
    }

    package func item(_ id: UUID) -> StudioLibraryItem? {
        itemsByID[id]
    }

    /// The rows `id` was made from.
    package func madeFrom(_ id: UUID) -> [StudioLibraryItem] {
        (sources[id] ?? []).compactMap { itemsByID[$0] }
    }

    /// The rows that read `id`'s output, newest first.
    package func usedIn(_ id: UUID) -> [StudioLibraryItem] {
        (uses[id] ?? []).compactMap { itemsByID[$0] }
    }

    package func hasLinks(_ id: UUID) -> Bool {
        sources[id] != nil || uses[id] != nil
    }

    /// The sources a run about to be recorded read: the rows, other than itself, whose outputs
    /// its input paths name. Used when a run is submitted.
    package static func sources(for item: StudioLibraryItem, in items: [StudioLibraryItem]) -> [UUID] {
        inferredSources(for: item, producers: producers(in: items.filter { !$0.isConversation && $0.id != item.id }))
    }

    /// Every file a run read: the recorded input, then each positional argument and option the
    /// command's contract declares as an input file, in argv order. An output destination is never
    /// an input, so a run that overwrites an earlier run's file is not made from it.
    package static func inputPaths(of item: StudioLibraryItem) -> [String] {
        var paths: [String] = []
        if let inputURL = item.inputURL { paths.append(inputURL.path) }
        if let arguments = item.commandArguments, let capability = item.templateID?.capability {
            let form = StudioConsoleCommand.seed(capability: capability, arguments: arguments)
            for (index, argument) in capability.arguments.enumerated() where argument.kind == .file {
                let values = argument.repeatable ? Array(form.arguments.dropFirst(index)) : Array(form.arguments.dropFirst(index).prefix(1))
                paths += values
            }
            for option in capability.options where isInputFile(option, of: capability) {
                paths += StudioAttachmentSlot.separatedPaths(form.text(option.flag))
            }
        }
        var seen = Set<String>()
        return paths
            .map { NSString(string: $0.trimmingCharacters(in: .whitespacesAndNewlines)).expandingTildeInPath }
            .filter { $0.hasPrefix("/") }
            .map { URL(fileURLWithPath: $0).standardizedFileURL.path }
            .filter { seen.insert($0).inserted }
    }

    /// A file option the run reads rather than writes: not the contract's destination, not in its
    /// Output section, and not one of the sidecar destinations named `--…-output`.
    private static func isInputFile(_ option: MereRunCapabilityOption, of capability: MereRunCommandCapability) -> Bool {
        option.kind == .file
            && option.flag != capability.output.flag
            && option.group != MereRunCapabilityOptionGroup.output
            && !option.flag.hasSuffix("-output")
    }

    /// Each output path, with the rows that wrote it oldest first: a path several runs wrote in
    /// turn belongs, for any later run, to the last one before it.
    private static func producers(in items: [StudioLibraryItem]) -> [String: [StudioLibraryItem]] {
        var producers: [String: [StudioLibraryItem]] = [:]
        for item in items where item.status == .completed {
            for url in item.allArtifactURLs {
                producers[url.standardizedFileURL.path, default: []].append(item)
            }
        }
        return producers.mapValues { $0.sorted { $0.createdAt < $1.createdAt } }
    }

    private static func inferredSources(for item: StudioLibraryItem, producers: [String: [StudioLibraryItem]]) -> [UUID] {
        var found: [UUID] = []
        for path in inputPaths(of: item) {
            guard let producer = producers[path]?.last(where: { $0.id != item.id && $0.createdAt < item.createdAt }),
                  !found.contains(producer.id) else { continue }
            found.append(producer.id)
        }
        return found
    }
}

extension StudioLibraryStore {
    /// The Library's lineage, built the first time it is read after the rows change.
    package var lineage: StudioLibraryLineage {
        if let lineageCache { return lineageCache }
        let built = StudioLibraryLineage(items: items)
        lineageCache = built
        return built
    }

    /// Takes one "Made from" link off a row, one undo step. An inferred link is written down as
    /// the row's remaining sources, so it does not come back the next time the index is built.
    package func removeSource(_ sourceID: UUID, from itemID: UUID) {
        let current = lineage.sources[itemID] ?? []
        guard current.contains(sourceID) else { return }
        setSources(current.filter { $0 != sourceID }, of: itemID, undoName: "Remove Link")
    }

    /// Writes a row's recorded sources, registering the write that puts the previous ones back.
    package func setSources(_ sourceIDs: [UUID]?, of itemID: UUID, undoName: String) {
        guard var item = items.first(where: { $0.id == itemID }) else { return }
        let previous = item.sourceItemIDs
        guard previous != sourceIDs else { return }
        item.sourceItemIDs = sourceIDs
        upsert(item)
        undo.register(undoName) { [weak self] in self?.setSources(previous, of: itemID, undoName: undoName) }
    }
}
