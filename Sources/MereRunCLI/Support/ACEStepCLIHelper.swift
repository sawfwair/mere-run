import ArgumentParser
import AudioCodecs
import Foundation
import MLX
import MereRunCore

enum ACEStepCLIHelper {
    static func resolveUserPath(_ path: String) -> URL {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == "~" {
            return URL(fileURLWithPath: NSHomeDirectory()).standardizedFileURL
        }
        if trimmed.hasPrefix("~/") {
            let suffix = String(trimmed.dropFirst(2))
            return URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent(suffix)
                .standardizedFileURL
        }
        return URL(fileURLWithPath: trimmed).standardizedFileURL
    }

    static func loadAudio48kHz(_ path: String, label: String) throws -> MLXArray {
        let url = resolveUserPath(path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("\(label) not found: \(url.path)")
        }

        let buffer = try AudioReader.readAudioBuffer(from: url, sampleRate: 48_000, channels: 2)
        guard buffer.isInterleaved else {
            throw ValidationError("\(label) decoded to a non-interleaved buffer: \(url.path)")
        }
        guard buffer.channelCount == 2 else {
            throw ValidationError("\(label) must decode to stereo at 48 kHz; got \(buffer.channelCount) channel(s).")
        }
        guard buffer.samples.count >= buffer.channelCount else {
            throw ValidationError("\(label) contains no audio samples: \(url.path)")
        }

        let frames = buffer.samples.count / buffer.channelCount
        let trimmedSampleCount = frames * buffer.channelCount
        let samples = trimmedSampleCount == buffer.samples.count
            ? buffer.samples
            : Array(buffer.samples.prefix(trimmedSampleCount))
        let clamped = samples.map { sample -> Float in
            guard sample.isFinite else {
                return 0
            }
            return max(-1.0, min(1.0, sample))
        }
        return MLXArray(clamped, [1, frames, buffer.channelCount]).asType(.float32)
    }

    static func durationSeconds(of audio48kHz: MLXArray, fallback: Float) -> Float {
        guard audio48kHz.ndim >= 2 else {
            return fallback
        }
        let frames = audio48kHz.dim(1)
        guard frames > 0 else {
            return fallback
        }
        return Float(frames) / 48_000.0
    }

    static func resolveCheckpointsRoot(
        model: String,
        checkpointsRoot: String?,
        turboSubdirectory: String,
        vaeSubdirectory: String,
        lmSubdirectory: String?,
        textSubdirectory: String?
    ) async throws -> URL {
        let candidates = buildCheckpointCandidates(
            model: model,
            checkpointsRoot: checkpointsRoot
        )

        for candidate in candidates {
            if isUsableCheckpointsRoot(
                candidate,
                turboSubdirectory: turboSubdirectory,
                vaeSubdirectory: vaeSubdirectory,
                lmSubdirectory: lmSubdirectory,
                textSubdirectory: textSubdirectory
            ) {
                return candidate
            }
            let nested = candidate.appendingPathComponent("checkpoints", isDirectory: true)
            if isUsableCheckpointsRoot(
                nested,
                turboSubdirectory: turboSubdirectory,
                vaeSubdirectory: vaeSubdirectory,
                lmSubdirectory: lmSubdirectory,
                textSubdirectory: textSubdirectory
            ) {
                return nested
            }
        }

        if let explicit = checkpointsRoot?.trimmingCharacters(in: .whitespacesAndNewlines), !explicit.isEmpty {
            throw ValidationError("Checkpoints root not found or incomplete: \(explicit)")
        }

        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: model,
                defaultModelID: ModelResolver.ModelID.aceStep.rawValue
            )
            let root = resolved.url
            if isUsableCheckpointsRoot(
                root,
                turboSubdirectory: turboSubdirectory,
                vaeSubdirectory: vaeSubdirectory,
                lmSubdirectory: lmSubdirectory,
                textSubdirectory: textSubdirectory
            ) {
                return root
            }
            let nested = root.appendingPathComponent("checkpoints", isDirectory: true)
            if isUsableCheckpointsRoot(
                nested,
                turboSubdirectory: turboSubdirectory,
                vaeSubdirectory: vaeSubdirectory,
                lmSubdirectory: lmSubdirectory,
                textSubdirectory: textSubdirectory
            ) {
                return nested
            }
        } catch let error as ManagedModelResolver.ResolverError {
            throw ValidationError(error.localizedDescription)
        }

        throw ValidationError("Music Acestep checkpoints not found. Add --checkpoints-root or set MERERUN_MUSIC_ACESTEP_ROOT.")
    }

    static func resolveTurboSubdirectory(at root: URL, explicit: String) throws -> String {
        let fm = FileManager.default

        let trimmed = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
        let upstreamDefault = "acestep-v15-turbo"
        let compatibilityDefault = "music-acestep-v15-turbo"
        let xlTurboDefault = "acestep-v15-xl-turbo"
        let candidates = trimmed == upstreamDefault
            ? [upstreamDefault, compatibilityDefault, xlTurboDefault]
            : [trimmed]

        for candidate in candidates where isUsableTurboDirectory(
            root.appendingPathComponent(candidate, isDirectory: true),
            fileManager: fm
        ) {
            return candidate
        }
        throw ValidationError("--turbo-subdirectory not found: \(trimmed)")
    }

    static func resolveLMSubdirectory(at root: URL, explicit: String?) throws -> String? {
        let fm = FileManager.default

        if let explicit, !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let explicitNormalized = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            let explicitRoot = root.appendingPathComponent(explicitNormalized, isDirectory: true)
            guard isUsableLMDirectory(explicitRoot, fileManager: fm) else {
                throw ValidationError("--lm-subdirectory not found: \(explicitNormalized)")
            }
            return explicitNormalized
        }

        let preferredCandidates = [
            "acestep-5Hz-lm-1.7B",
            "acestep-5hz-lm-1.7b",
            "acestep-5Hz-lm-4B",
            "acestep-5hz-lm-4b",
            "acestep-5Hz-lm",
            "acestep-5hz-lm",
            "music-acestep-5hz-lm-1.7b",
            "music-acestep-5Hz-lm-1.7B",
            "music-acestep-5hz-lm-4b",
            "music-acestep-5Hz-lm-4B",
            "music-acestep-5hz-lm",
            "music-acestep-5Hz-lm",
            "lm",
            "music-acestep-lm"
        ]

        for candidate in preferredCandidates {
            if isUsableLMDirectory(root.appendingPathComponent(candidate, isDirectory: true), fileManager: fm) {
                return candidate
            }
        }

        let discovered = (try? fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ))?.first(where: { directory in
            let isDirectory = (try? directory.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
            let name = directory.lastPathComponent.lowercased()
            return isDirectory
                && name.contains("lm")
                && name.contains("5hz")
                && isUsableLMDirectory(directory, fileManager: fm)
        })

        return discovered?.lastPathComponent
    }

    static func resolveTextSubdirectory(at root: URL, explicit: String?) throws -> String? {
        let fm = FileManager.default

        if let explicit, !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let explicitNormalized = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            let explicitRoot = root.appendingPathComponent(explicitNormalized, isDirectory: true)
            guard isDirectory(explicitRoot, fileManager: fm) else {
                throw ValidationError("--text-subdirectory not found: \(explicitNormalized)")
            }
            return explicitNormalized
        }

        let preferredCandidates = [
            "Qwen3-Embedding-0.6B",
            "qwen3-embedding-0.6b",
            "Qwen3-Embedding-4B",
            "qwen3-embedding-4b",
            "Qwen3-Embedding",
            "qwen3-embedding",
            "text_encoder",
            "text-encoder"
        ]

        for candidate in preferredCandidates {
            if isDirectory(root.appendingPathComponent(candidate, isDirectory: true), fileManager: fm) {
                return candidate
            }
        }

        let discovered = (try? fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ))?.first(where: { directory in
            let isDirectory = (try? directory.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
            let name = directory.lastPathComponent.lowercased()
            return isDirectory && (
                (name.contains("qwen3") && name.contains("embedding"))
                || name == "text_encoder"
                || name == "text-encoder"
            )
        })

        return discovered?.lastPathComponent
    }

    static func buildCheckpointCandidates(
        model: String,
        checkpointsRoot: String?
    ) -> [URL] {
        var candidates: [URL] = []

        if let explicit = checkpointsRoot?.trimmingCharacters(in: .whitespacesAndNewlines), !explicit.isEmpty {
            candidates.append(URL(fileURLWithPath: explicit).standardizedFileURL)
        }

        var modelPathExists = false
        let explicitModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        if !explicitModel.isEmpty {
            let url = URL(fileURLWithPath: explicitModel).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                modelPathExists = true
                candidates.append(url)
            }
        }

        if let envRoot = ProcessInfo.processInfo.environment["MERERUN_MUSIC_ACESTEP_ROOT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !envRoot.isEmpty
        {
            candidates.append(URL(fileURLWithPath: envRoot).standardizedFileURL)
        }

        let managedModelID = explicitModel.lowercased()
        if !managedModelID.isEmpty && !modelPathExists {
            let requestedRoot = MereRunModelPaths.modelsDir
                .appendingPathComponent(managedModelID, isDirectory: true)
                .standardizedFileURL
            candidates.append(requestedRoot)
            candidates.append(requestedRoot.appendingPathComponent("checkpoints", isDirectory: true))
        }

        if managedModelID.isEmpty {
            let localModelRoot = MereRunModelPaths.modelsDir
                .appendingPathComponent(ModelResolver.ModelID.aceStep.rawValue, isDirectory: true)
                .standardizedFileURL
            candidates.append(localModelRoot)
            candidates.append(localModelRoot.appendingPathComponent("checkpoints", isDirectory: true))
        }

        return candidates
    }

    private static func isUsableCheckpointsRoot(
        _ root: URL,
        turboSubdirectory: String,
        vaeSubdirectory: String,
        lmSubdirectory: String?,
        textSubdirectory: String?
    ) -> Bool {
        let fm = FileManager.default

        guard isDirectory(root, fileManager: fm) else {
            return false
        }
        let turboCandidates = turboSubdirectory == "acestep-v15-turbo"
            ? ["acestep-v15-turbo", "music-acestep-v15-turbo", "acestep-v15-xl-turbo"]
            : [turboSubdirectory]
        guard turboCandidates.contains(where: {
            isUsableTurboDirectory(root.appendingPathComponent($0, isDirectory: true), fileManager: fm)
        }) else {
            return false
        }
        guard isDirectory(root.appendingPathComponent(vaeSubdirectory), fileManager: fm) else {
            return false
        }

        if let lmSubdirectory, !lmSubdirectory.isEmpty {
            guard isDirectory(root.appendingPathComponent(lmSubdirectory), fileManager: fm) else {
                return false
            }
        }

        if let textSubdirectory, !textSubdirectory.isEmpty {
            guard isDirectory(root.appendingPathComponent(textSubdirectory), fileManager: fm) else {
                return false
            }
        }

        return true
    }

    private static func isUsableTurboDirectory(_ url: URL, fileManager: FileManager) -> Bool {
        isDirectory(url, fileManager: fileManager)
            && ACEStepResources(rootURL: url).validate(fileManager: fileManager).isEmpty
    }

    private static func isUsableLMDirectory(_ url: URL, fileManager: FileManager) -> Bool {
        isDirectory(url, fileManager: fileManager)
            && ACEStep5HzLMResources(rootURL: url).validate(fileManager: fileManager).isEmpty
    }

    private static func isDirectory(_ url: URL, fileManager: FileManager) -> Bool {
        var isDir: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
    }
}
