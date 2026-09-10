import Foundation
import AudioCore

public typealias ParakeetExecutionProvider = AudioCore.ParakeetExecutionProvider

/// Resolves Core ML artifacts that bundle a complete Parakeet model.
extension ParakeetExecutionProvider {
    public var bundledModelURL: URL? {
        guard case .coreML(let artifactURL) = self else { return nil }
        let root = artifactURL.standardizedFileURL
        return ParakeetResources(rootURL: root).validate().isEmpty ? root : nil
    }
}
