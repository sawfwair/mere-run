import Foundation

enum FileSystemHelper {
    /// Recursively compute the total size of all regular files under a directory.
    static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        let root = url.resolvingSymlinksInPath()
        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                  values.isRegularFile == true,
                  let size = values.fileSize else { continue }
            total += Int64(size)
        }
        return total
    }

    /// Copy a directory while resolving symlinks (some model stores use symlinked blobs).
    static func copyDirectoryResolvingSymlinks(from source: URL, to destination: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)

        let sourceValues = try source.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
        let resolvedSource = sourceValues.isSymbolicLink == true ? source.resolvingSymlinksInPath() : source

        let contents = try fm.contentsOfDirectory(at: resolvedSource, includingPropertiesForKeys: [.isSymbolicLinkKey, .isDirectoryKey])
        for item in contents {
            let destItem = destination.appendingPathComponent(item.lastPathComponent)
            let resourceValues = try item.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])

            if resourceValues.isSymbolicLink == true {
                let resolved = item.resolvingSymlinksInPath()
                let resolvedValues = try resolved.resourceValues(forKeys: [.isDirectoryKey])
                if resolvedValues.isDirectory == true {
                    try copyDirectoryResolvingSymlinks(from: resolved, to: destItem)
                } else {
                    let data = try Data(contentsOf: item)
                    try data.write(to: destItem)
                }
            } else if resourceValues.isDirectory == true {
                try copyDirectoryResolvingSymlinks(from: item, to: destItem)
            } else {
                let data = try Data(contentsOf: item)
                try data.write(to: destItem)
            }
        }
    }
}
