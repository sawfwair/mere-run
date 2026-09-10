import Foundation

/// Selects the execution engine for the Parakeet encoder. AudioSTT resolves
/// and validates the model assets for the selected provider.
public enum ParakeetExecutionProvider: Codable, Sendable, Hashable {
    case mlx
    case coreML(artifactURL: URL)
}
