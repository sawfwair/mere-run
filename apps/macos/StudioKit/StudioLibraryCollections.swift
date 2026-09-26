import Foundation

// Collections: named sets of Library rows the user gathers by hand. A row can sit in several;
// deleting a collection forgets only the set, never its rows.
//
// They live in `collections.json` beside `library.json` rather than in it. The library file is a
// top-level array every earlier build reads, and giving it a wrapper object would make those
// builds recover it as corrupt; a sibling file they never open keeps both directions safe. The
// membership is kept on the collection, so a build that rewrites `library.json` without knowing
// about collections leaves them whole.

/// One collection: its name and its rows, in the order they were added.
package struct StudioLibraryCollection: Codable, Identifiable, Equatable {
    package let id: UUID
    package var name: String
    package var createdAt: Date
    package var itemIDs: [UUID]

    package init(id: UUID = UUID(), name: String, createdAt: Date = Date(), itemIDs: [UUID] = []) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.itemIDs = itemIDs
    }

    package func contains(_ itemID: UUID) -> Bool {
        itemIDs.contains(itemID)
    }
}

/// `collections.json`: a versioned object, so a later shape can be told apart.
package struct StudioLibraryCollectionsFile: Codable, Equatable {
    package static let currentVersion = 1

    package var version: Int
    package var collections: [StudioLibraryCollection]

    package init(version: Int = Self.currentVersion, collections: [StudioLibraryCollection]) {
        self.version = version
        self.collections = collections
    }
}

extension StudioLibraryStore {
    /// Where collections are kept: beside the library file, so a test's temporary Library keeps
    /// its collections in the same temporary folder.
    package var collectionsURL: URL {
        libraryURL.deletingLastPathComponent().appendingPathComponent("collections.json", isDirectory: false)
    }

    /// The name a new collection takes when the user gives none: "New collection", then
    /// "New collection 2", and so on past every name in use.
    package var suggestedCollectionName: String {
        let base = "New collection"
        let names = Set(collections.map { $0.name.lowercased() })
        guard names.contains(base.lowercased()) else { return base }
        var number = 2
        while names.contains("\(base) \(number)".lowercased()) { number += 1 }
        return "\(base) \(number)"
    }

    /// The collections `itemID` belongs to, in the column's order.
    package func collections(containing itemID: UUID) -> [StudioLibraryCollection] {
        collections.filter { $0.contains(itemID) }
    }

    /// The rows of `collection` the Library still holds; a row deleted since is left out (and
    /// comes back with it on Undo).
    package func memberCount(of collection: StudioLibraryCollection) -> Int {
        let present = Set(items.map(\.id))
        return collection.itemIDs.filter(present.contains).count
    }

    /// Creates a collection holding `itemIDs`, one undo step. A blank name takes the suggested one.
    @discardableResult
    package func createCollection(named name: String, adding itemIDs: [UUID] = []) -> StudioLibraryCollection {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let collection = StudioLibraryCollection(
            name: trimmed.isEmpty ? suggestedCollectionName : trimmed,
            itemIDs: Self.unique(itemIDs)
        )
        insertCollection(collection, at: collections.count, undoName: "New Collection")
        return collection
    }

    package func renameCollection(id: UUID, to name: String) {
        guard let index = collections.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let previous = collections[index].name
        guard !trimmed.isEmpty, trimmed != previous else { return }
        collections[index].name = trimmed
        saveCollections()
        undo.register("Rename Collection") { [weak self] in self?.renameCollection(id: id, to: previous) }
    }

    /// Forgets the collection. Its rows stay in the Library, and in any other collection.
    package func deleteCollection(id: UUID) {
        guard let index = collections.firstIndex(where: { $0.id == id }) else { return }
        let removed = collections.remove(at: index)
        saveCollections()
        undo.register("Delete Collection") { [weak self] in
            self?.insertCollection(removed, at: index, undoName: "Delete Collection")
        }
    }

    /// Adds the rows not already in the collection, one step; adding what is already there is not
    /// a step at all.
    package func addToCollection(id: UUID, itemIDs: [UUID]) {
        guard let index = collections.firstIndex(where: { $0.id == id }) else { return }
        let added = Self.unique(itemIDs).filter { !collections[index].contains($0) }
        guard !added.isEmpty else { return }
        collections[index].itemIDs += added
        saveCollections()
        undo.register("Add to Collection") { [weak self] in self?.removeFromCollection(id: id, itemIDs: added) }
    }

    /// Takes the rows out of the collection; the rows themselves stay in the Library.
    package func removeFromCollection(id: UUID, itemIDs: [UUID]) {
        guard let index = collections.firstIndex(where: { $0.id == id }) else { return }
        let removing = Set(itemIDs)
        let removed = collections[index].itemIDs.filter(removing.contains)
        guard !removed.isEmpty else { return }
        collections[index].itemIDs.removeAll(where: removing.contains)
        saveCollections()
        undo.register("Remove from Collection") { [weak self] in self?.addToCollection(id: id, itemIDs: removed) }
    }

    /// Adds `itemIDs` to the collection when any of them is missing from it, otherwise takes them
    /// all out: the check mark in a row's Add to collection menu.
    package func toggleMembership(collectionID: UUID, itemIDs: [UUID]) {
        guard let collection = collections.first(where: { $0.id == collectionID }) else { return }
        if itemIDs.allSatisfy(collection.contains) {
            removeFromCollection(id: collectionID, itemIDs: itemIDs)
        } else {
            addToCollection(id: collectionID, itemIDs: itemIDs)
        }
    }

    /// The rows a dropped file belongs to: every run that made it.
    package func itemIDs(producing urls: [URL]) -> [UUID] {
        let paths = Set(urls.map(\.standardizedFileURL.path))
        return items.filter { item in
            item.allArtifactURLs.contains { paths.contains($0.standardizedFileURL.path) }
        }.map(\.id)
    }

    private func insertCollection(_ collection: StudioLibraryCollection, at index: Int, undoName: String) {
        collections.insert(collection, at: min(index, collections.count))
        saveCollections()
        undo.register(undoName) { [weak self] in self?.deleteCollection(id: collection.id) }
    }

    /// Reads `collections.json`. A missing file is no collections; a file this build cannot read
    /// is moved aside, like a corrupt library, so the next save never overwrites it.
    func loadCollections() {
        guard fileManager.fileExists(atPath: collectionsURL.path) else {
            collections = []
            return
        }
        do {
            let file = try JSONDecoder.mereRunApp.decode(
                StudioLibraryCollectionsFile.self,
                from: Data(contentsOf: collectionsURL)
            )
            collections = file.collections
        } catch {
            collections = []
            try? fileManager.moveItem(at: collectionsURL, to: Self.siblingURL(of: collectionsURL, tag: "corrupt"))
        }
    }

    func saveCollections() {
        do {
            try fileManager.createDirectory(at: collectionsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try JSONEncoder.mereRunApp.encode(StudioLibraryCollectionsFile(collections: collections))
            try data.write(to: collectionsURL, options: [.atomic])
            lastPersistenceError = nil
        } catch {
            lastPersistenceError = error.localizedDescription
        }
    }

    private static func unique(_ ids: [UUID]) -> [UUID] {
        var seen = Set<UUID>()
        return ids.filter { seen.insert($0).inserted }
    }
}
