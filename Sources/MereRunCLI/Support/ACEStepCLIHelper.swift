import ArgumentParser
import Foundation
import MLX
import MereRunCore

/// Preserves command validation diagnostics over shared runtime preparation.
enum ACEStepCLIHelper {
    typealias LMResolution = ACEStepRuntimePreparation.LMResolution

    static func resolveUserPath(_ path: String) -> URL {
        ACEStepRuntimePreparation.resolveUserPath(path)
    }

    static func loadAudio48kHz(_ path: String, label: String) throws -> MLXArray {
        do {
            return try ACEStepRuntimePreparation.loadAudio48kHz(path, label: label)
        } catch let error as ACEStepPreparationIssue {
            throw ValidationError(error.message)
        }
    }

    static func durationSeconds(of audio48kHz: MLXArray, fallback: Float) -> Float {
        ACEStepRuntimePreparation.durationSeconds(of: audio48kHz, fallback: fallback)
    }

    static func resolveCheckpointsRoot(
        model: String,
        checkpointsRoot: String?,
        turboSubdirectory: String,
        vaeSubdirectory: String,
        lmSubdirectory: String?,
        textSubdirectory: String?
    ) async throws -> URL {
        do {
            return try await ACEStepRuntimePreparation.resolveCheckpointsRoot(model: model, checkpointsRoot: checkpointsRoot, turboSubdirectory: turboSubdirectory, vaeSubdirectory: vaeSubdirectory, lmSubdirectory: lmSubdirectory, textSubdirectory: textSubdirectory)
        } catch let error as ACEStepPreparationIssue {
            throw ValidationError(error.message)
        }
    }

    static func resolveTurboSubdirectory(at root: URL, explicit: String) throws -> String {
        do {
            return try ACEStepRuntimePreparation.resolveTurboSubdirectory(at: root, explicit: explicit)
        } catch let error as ACEStepPreparationIssue {
            throw ValidationError(error.message)
        }
    }

    static func resolveLMSubdirectory(at root: URL, explicit: String?) throws -> String? {
        do {
            return try ACEStepRuntimePreparation.resolveLMSubdirectory(at: root, explicit: explicit)
        } catch let error as ACEStepPreparationIssue {
            throw ValidationError(error.message)
        }
    }

    static func resolveLMResources(
        checkpointsRoot: URL,
        lmModel: String?,
        lmSubdirectory: String?
    ) async throws -> LMResolution {
        do {
            return try await ACEStepRuntimePreparation.resolveLMResources(checkpointsRoot: checkpointsRoot, lmModel: lmModel, lmSubdirectory: lmSubdirectory)
        } catch let error as ACEStepPreparationIssue {
            throw ValidationError(error.message)
        }
    }

    static func resolveLMRoot(at root: URL) -> URL? {
        ACEStepRuntimePreparation.resolveLMRoot(at: root)
    }

    static func resolveTextSubdirectory(at root: URL, explicit: String?) throws -> String? {
        do {
            return try ACEStepRuntimePreparation.resolveTextSubdirectory(at: root, explicit: explicit)
        } catch let error as ACEStepPreparationIssue {
            throw ValidationError(error.message)
        }
    }

    static func buildCheckpointCandidates(
        model: String,
        checkpointsRoot: String?
    ) -> [URL] {
        ACEStepRuntimePreparation.buildCheckpointCandidates(model: model, checkpointsRoot: checkpointsRoot)
    }

}
