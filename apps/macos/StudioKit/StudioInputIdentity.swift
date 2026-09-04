import Foundation

/// File metadata lets Analyze detect a replaced file even when its path stays the same.
package struct StudioInputIdentity: Codable, Equatable {
    package let size: Int
    package let modifiedAt: TimeInterval

    package static func read(_ url: URL) -> Self? {
        var fresh = URL(fileURLWithPath: url.path)
        fresh.removeAllCachedResourceValues()
        guard let values = try? fresh.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]),
              let size = values.fileSize, let date = values.contentModificationDate else { return nil }
        return Self(size: size, modifiedAt: date.timeIntervalSince1970)
    }

    package static func matches(item: StudioLibraryItem, input: URL?) -> Bool {
        guard let recorded = item.inputURL, let input,
              recorded.resolvingSymlinksInPath().standardizedFileURL == input.resolvingSymlinksInPath().standardizedFileURL else { return false }
        guard let identity = item.inputIdentity else { return true }
        return read(input) == identity
    }
}
