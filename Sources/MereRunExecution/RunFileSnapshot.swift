import Foundation

/// Records a metadata file, including its absence, for later change detection.
public struct RunFileSnapshot: Codable, Hashable, Sendable {
    public let url: URL
    public let artifact: RunArtifact?

    public static func capture(_ url: URL) throws -> Self {
        let artifact = FileManager.default.fileExists(atPath: url.path) ? try RunArtifact.read(url) : nil
        return Self(url: url, artifact: artifact)
    }

    public func isUnchanged() throws -> Bool {
        try Self.capture(url) == self
    }
}
