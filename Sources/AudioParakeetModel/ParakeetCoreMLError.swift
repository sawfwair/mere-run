import Foundation

package enum ParakeetCoreMLError: LocalizedError {
    case missingManifest(String)
    case missingCompiledModel(String)
    case missingCompiledDecoder(String)
    case unsupportedSchemaVersion(Int)
    case untrustedSource(repository: String, revision: String, license: String)
    case untrustedConversion
    case incompatibleManifest
    case emptyArtifactClosure
    case unsafeArtifactPath(String)
    case artifactClosureMismatch(expected: [String], actual: [String])
    case symlinkedArtifact(String)
    case unsupportedArtifact(String)
    case unsupportedInputShape([Int])
    case unsupportedDecoderInputShape([Int])
    case inputTooLong(actual: Int, maximum: Int)
    case invalidEmbeddingByteCount(expected: Int, actual: Int)
    case missingOutput(String)
    case unsupportedMultiArrayType(String)
    case invalidOutputShape([Int])

    package var errorDescription: String? {
        switch self {
        case .missingManifest(let path):
            return "The Parakeet Core ML manifest is missing: \(path)"
        case .missingCompiledModel(let path):
            return "The compiled Parakeet Core ML encoder is missing: \(path)"
        case .missingCompiledDecoder(let path):
            return "The compiled Parakeet Core ML decoder is missing: \(path)"
        case .unsupportedSchemaVersion(let version):
            return "Unsupported Parakeet Core ML manifest schema version: \(version)."
        case .untrustedSource(let repository, let revision, let license):
            return "Untrusted Parakeet Core ML source: \(repository)@\(revision) (\(license))."
        case .untrustedConversion:
            return "The Parakeet Core ML artifact was not built with Mere's pinned conversion environment."
        case .incompatibleManifest:
            return "The Parakeet Core ML artifact is incompatible with the selected decoder checkpoint."
        case .emptyArtifactClosure:
            return "The Parakeet Core ML manifest does not pin any compiled artifacts."
        case .unsafeArtifactPath(let path):
            return "Unsafe path in the Parakeet Core ML manifest: \(path)"
        case .artifactClosureMismatch(let expected, let actual):
            return "The Parakeet Core ML artifact closure differs from its manifest: expected \(expected), found \(actual)."
        case .symlinkedArtifact(let path):
            return "The Parakeet Core ML artifact contains a symbolic link: \(path)"
        case .unsupportedArtifact(let path):
            return "The Parakeet Core ML artifact contains an unsupported file type: \(path)"
        case .unsupportedInputShape(let shape):
            return "The Parakeet Core ML encoder requires [1, frames, features]; found \(shape)."
        case .unsupportedDecoderInputShape(let shape):
            return "The Parakeet Core ML decoder received an unsupported encoder shape: \(shape)."
        case .inputTooLong(let actual, let maximum):
            return "The Parakeet Core ML encoder accepts at most \(maximum) mel frames; found \(actual)."
        case .invalidEmbeddingByteCount(let expected, let actual):
            return "The Parakeet Core ML decoder embedding table requires \(expected) bytes; found \(actual)."
        case .missingOutput(let name):
            return "The Parakeet Core ML encoder did not return \(name)."
        case .unsupportedMultiArrayType(let type):
            return "Unsupported Core ML tensor type: \(type)."
        case .invalidOutputShape(let shape):
            return "The Parakeet Core ML encoder returned an invalid shape: \(shape)."
        }
    }
}
