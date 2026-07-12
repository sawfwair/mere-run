import Foundation

public struct Wan2DreamXCausalResources: Hashable, Sendable {
    public static let modelID = "video-dreamx-world-5b-ar-mlx"
    public static let upstreamRepoID = "GD-ML/DreamX-World-5B"
    public static let upstreamRevision = "67487c4a61466bb7166d30b7187dd465e0ac9f6c"
    public static let weightsFilename = "model.safetensors"

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var weightsURL: URL {
        rootURL.appendingPathComponent(Self.weightsFilename)
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        fileManager.fileExists(atPath: weightsURL.path) ? [] : [weightsURL]
    }
}
