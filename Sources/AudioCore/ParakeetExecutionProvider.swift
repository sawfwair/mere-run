import Foundation

/// Selects the speech audio encoder engine (Parakeet or SOV). AudioSTT resolves
/// and validates the model assets for the selected provider.
public enum ParakeetExecutionProvider: Codable, Sendable, Hashable {
    case mlx
    case coreML(artifactURL: URL)
}
