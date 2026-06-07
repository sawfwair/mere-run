import Foundation

enum FileSystemHelper {
    struct DirectoryUsage: Equatable {
        let resolvedBytes: Int64
        let localBytes: Int64
        let rootIsSymlink: Bool
        let symlinkCount: Int
        let symlinkedDirectoryCount: Int
        let symlinkedFileCount: Int

        var layoutDescription: String {
            if rootIsSymlink, symlinkCount > 0 {
                return "symlinked root with symlinked entries"
            }
            if rootIsSymlink {
                return "symlinked root"
            }
            if symlinkedDirectoryCount > 0, symlinkedFileCount > 0 {
                return "directory with symlinked entries"
            }
            if symlinkedDirectoryCount > 0 {
                return "directory with symlinked directories"
            }
            if symlinkedFileCount > 0 {
                return "directory with symlinked files"
            }
            if symlinkCount > 0 {
                return "directory with symlinks"
            }
            return "direct directory"
        }
    }

    /// Recursively compute the total size of all regular files under a directory.
    static func directorySize(at url: URL) -> Int64 {
        directoryUsage(at: url).resolvedBytes
    }

    /// Recursively inspect a directory, following symlinked model payload files and directories.
    static func directoryUsage(at url: URL) -> DirectoryUsage {
        DirectoryUsageScanner(fileManager: .default).usage(at: url)
    }

    private final class DirectoryUsageScanner {
        private let fileManager: FileManager
        private var resolvedBytes: Int64 = 0
        private var localBytes: Int64 = 0
        private var symlinkCount = 0
        private var symlinkedDirectoryCount = 0
        private var symlinkedFileCount = 0
        private var visitedDirectories: Set<String> = []
        private var visitedFiles: Set<String> = []

        init(fileManager: FileManager) {
            self.fileManager = fileManager
        }

        func usage(at url: URL) -> DirectoryUsage {
            let rootIsSymlink = (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true
            let root = rootIsSymlink ? url.resolvingSymlinksInPath() : url
            scan(root, reachedThroughSymlink: rootIsSymlink)
            return DirectoryUsage(
                resolvedBytes: resolvedBytes,
                localBytes: localBytes,
                rootIsSymlink: rootIsSymlink,
                symlinkCount: symlinkCount,
                symlinkedDirectoryCount: symlinkedDirectoryCount,
                symlinkedFileCount: symlinkedFileCount
            )
        }

        private func scan(_ url: URL, reachedThroughSymlink: Bool) {
            guard let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .fileSizeKey]) else {
                return
            }

            if values.isRegularFile == true {
                addFile(url, reachedThroughSymlink: reachedThroughSymlink)
                return
            }

            guard values.isDirectory == true else { return }
            let directoryKey = url.resolvingSymlinksInPath().standardizedFileURL.path
            guard visitedDirectories.insert(directoryKey).inserted else { return }

            let contents = (try? fileManager.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
            )) ?? []

            for item in contents {
                let itemIsSymlink = (try? item.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true
                if itemIsSymlink {
                    symlinkCount += 1
                    let resolved = item.resolvingSymlinksInPath()
                    guard let resolvedValues = try? resolved.resourceValues(
                        forKeys: [.isDirectoryKey, .isRegularFileKey, .fileSizeKey]
                    ) else {
                        continue
                    }
                    if resolvedValues.isDirectory == true {
                        symlinkedDirectoryCount += 1
                        scan(resolved, reachedThroughSymlink: true)
                    } else if resolvedValues.isRegularFile == true {
                        symlinkedFileCount += 1
                        addFile(resolved, reachedThroughSymlink: true)
                    }
                } else {
                    scan(item, reachedThroughSymlink: reachedThroughSymlink)
                }
            }
        }

        private func addFile(_ url: URL, reachedThroughSymlink: Bool) {
            let fileKey = url.resolvingSymlinksInPath().standardizedFileURL.path
            guard visitedFiles.insert(fileKey).inserted else { return }
            guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
                  let size = values.fileSize else { return }
            let byteCount = Int64(size)
            resolvedBytes += byteCount
            if !reachedThroughSymlink {
                localBytes += byteCount
            }
        }
    }

    /// Recursively compute the total size of all regular files under a directory without following symlinked children.
    static func localDirectorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        let root = url.resolvingSymlinksInPath()
        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let measuredValues = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                  measuredValues.isRegularFile == true,
                  let size = measuredValues.fileSize else { continue }
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
