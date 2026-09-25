import Foundation
import MereRunModelKit

/// Immutable source artifact for NVIDIA's released eight-speaker checkpoint.
/// The NeMo archive is an initializer container; inference runs in AudioSortformer.
public enum Nemotron3DiarizationResources {
    public static let modelID = ModelResolver.ModelID.nemotron3Diarization.rawValue
    public static let repository = "nvidia/Nemotron-3-Diarization"
    public static let revision = "723e19c601d99b7e58fba6a14e32153e0afe48d9"
    public static let archivePin = ModelArtifactPin(
        filename: "Nemotron-3-Diarization.nemo",
        byteCount: 198_676_480,
        sha256: "867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d"
    )
    public static let checkpointSHA256 = "9930e1719c85b35da083c557cdbd6503072c311ffe8bc9cbd3371d2609e21d0f"
    public static let configurationSHA256 = "0a738c13104761be11da4a315af0a94b3c14a1cd6509a3e788a78555751e1ccb"

    public static func verify(at rootURL: URL) throws -> URL {
        try archivePin.verify(in: rootURL)
    }

    /// Whether the speech diarization commands run Nemotron 3 for `model`, resolved to `root`:
    /// the managed id, or a folder that holds the NeMo archive. Anything else runs Sortformer.
    public static func isNemotron3(model: String, root: URL) -> Bool {
        model == modelID || FileManager.default.fileExists(atPath: root.appendingPathComponent(archivePin.filename).path)
    }
}
