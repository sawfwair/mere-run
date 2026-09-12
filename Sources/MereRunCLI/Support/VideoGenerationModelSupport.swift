import ArgumentParser
import Foundation
import MereRunContract
import MereRunCore

// These adapters preserve CLI validation diagnostics for the other video commands.
// Core owns checkpoint lookup and layout validation.
func resolveVideoModelRoot(
    explicitModelRoot: String?, requestedModel: String, variant: LTXVideoVariant,
    allowAutoDownload: Bool = true
) async throws -> URL {
    do {
        return try await VideoGenerationModelResolver.resolve(
            explicitModelRoot: explicitModelRoot, requestedModel: requestedModel,
            variant: variant, allowAutoDownload: allowAutoDownload
        )
    } catch VideoGenerationError.invalidInput(let message) {
        throw ValidationError(message)
    }
}

func validateNativeModelRoot(_ rootURL: URL) throws {
    do {
        try VideoGenerationModelResolver.validate(rootURL)
    } catch VideoGenerationError.invalidInput(let message) {
        throw ValidationError(message)
    }
}

func validateNativeAudioToVideoModelRoot(_ rootURL: URL, fileManager: FileManager = .default) throws {
    do {
        try VideoGenerationModelResolver.validateAudioToVideo(rootURL, fileManager: fileManager)
    } catch VideoGenerationError.invalidInput(let message) {
        throw ValidationError(message)
    }
}
