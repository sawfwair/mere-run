import Foundation

extension FileManager {
    /// Lists a directory after resolving the supplied path through any symlinked model-store
    /// components. Foundation does not traverse a directory when the final URL is a symlink.
    public func contentsOfDirectoryResolvingSymlinks(
        at url: URL,
        includingPropertiesForKeys keys: [URLResourceKey]? = nil,
        options mask: DirectoryEnumerationOptions = []
    ) throws -> [URL] {
        try contentsOfDirectory(
            at: url.resolvingSymlinksInPath(),
            includingPropertiesForKeys: keys,
            options: mask
        )
    }

    /// Creates a recursive enumerator after resolving the supplied root directory. Nested
    /// symlink children are still treated as entries, so callers retain control over recursion.
    public func enumeratorResolvingSymlinks(
        at url: URL,
        includingPropertiesForKeys keys: [URLResourceKey]? = nil,
        options mask: DirectoryEnumerationOptions = [],
        errorHandler handler: ((URL, any Error) -> Bool)? = nil
    ) -> DirectoryEnumerator? {
        enumerator(
            at: url.resolvingSymlinksInPath(),
            includingPropertiesForKeys: keys,
            options: mask,
            errorHandler: handler
        )
    }
}
