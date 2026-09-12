import Foundation
import MereRunContract

/// Resolves video checkpoints and validates their native layouts without loading tensors.
public enum VideoGenerationModelResolver {
    public static func validate(_ rootURL: URL) throws {
        let fm = FileManager.default
        let rootURL = rootURL.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: rootURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw VideoGenerationError.invalidInput("Model root directory not found: \(rootURL.path)")
        }

        let wanResources = Wan2Resources(rootURL: rootURL)
        if wanResources.validate().isEmpty, (try? wanResources.loadConfiguration()) != nil {
            return
        }

        if isLTX23AudioToVideoModelRoot(rootURL)
            || isLTX23SplitModelRoot(rootURL)
            || isLTX25FullModelRoot(rootURL)
            || isLTX25ModelRoot(rootURL) {
            return
        }

        let required = [
            rootURL.appendingPathComponent("text_encoder/config.json", isDirectory: false),
            rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json", isDirectory: false),
        ]
        for file in required where !fm.fileExists(atPath: file.path) {
            throw VideoGenerationError.invalidInput("Missing required LTX file: \(file.path)")
        }

        var tokenizerIsDir: ObjCBool = false
        let tokenizer = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        guard fm.fileExists(atPath: tokenizer.path, isDirectory: &tokenizerIsDir), tokenizerIsDir.boolValue else {
            throw VideoGenerationError.invalidInput("Missing tokenizer directory: \(tokenizer.path)")
        }

        let entries = (try? fm.contentsOfDirectoryResolvingSymlinks(
            at: rootURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        let hasTransformer = entries.contains { entry in
            let name = entry.lastPathComponent
            return name.hasPrefix("ltx-2-19") && name.hasSuffix(".safetensors")
        }
        let hasUpsampler = entries.contains { entry in
            let name = entry.lastPathComponent
            return name.hasPrefix("ltx-2-spatial-upscaler") && name.hasSuffix(".safetensors")
        }

        guard hasTransformer else {
            throw VideoGenerationError.invalidInput("Missing LTX transformer weights under \(rootURL.path)")
        }
        guard hasUpsampler else {
            throw VideoGenerationError.invalidInput("Missing LTX upsampler weights under \(rootURL.path)")
        }
    }

    public static func validateAudioToVideo(
        _ rootURL: URL,
        fileManager: FileManager = .default
    ) throws {
        let root = rootURL.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw VideoGenerationError.invalidInput("A2Vid model root directory not found: \(root.path)")
        }

        if isLTX25FullModelRoot(root, fileManager: fileManager) {
            return
        }

        let required = [
            "split_model.json",
            "config.json",
            "connector.safetensors",
            "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
        ]
        for relativePath in required {
            let file = root.appendingPathComponent(relativePath, isDirectory: false)
            guard fileManager.fileExists(atPath: file.path) else {
                throw VideoGenerationError.invalidInput(
                    "Missing required LTX 2.3 A2Vid file: \(file.path). Pull \(ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)."
                )
            }
        }
    }

    public static func resolve(
        explicitModelRoot: String?,
        requestedModel: String,
        variant: LTXVideoVariant,
        allowAutoDownload: Bool = true
    ) async throws -> URL {
        if let explicitModelRoot, !explicitModelRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: explicitModelRoot).standardizedFileURL
        }

        let trimmedModel = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedModel.isEmpty {
            let explicitModelURL = URL(fileURLWithPath: trimmedModel).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicitModelURL.path)
                || trimmedModel.lowercased() != ModelResolver.ModelID.ltxVideoAV.rawValue
            {
                if let modelID = ModelResolver.ModelID(rawValue: trimmedModel.lowercased()),
                   let installed = ModelResolver().resolveIfPresent(modelID) {
                    return installed.rootURL
                }
                do {
                    let resolved = try await ManagedModelResolver.resolveForRuntime(
                        requestedModel: trimmedModel,
                        defaultModelID: ModelResolver.ModelID.ltxVideoAV.rawValue,
                        allowAutoDownload: allowAutoDownload
                    )
                    return resolved.url
                } catch let error as ManagedModelResolver.ResolverError {
                    throw VideoGenerationError.invalidInput(error.localizedDescription)
                }
            }
        }

        if let suggested = suggestedVideoModelRoot(for: variant) {
            return URL(fileURLWithPath: suggested).standardizedFileURL
        }

        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: ModelResolver.ModelID.ltxVideoAV.rawValue,
                defaultModelID: ModelResolver.ModelID.ltxVideoAV.rawValue,
                allowAutoDownload: allowAutoDownload
            )
            return resolved.url
        } catch let error as ManagedModelResolver.ResolverError {
            throw VideoGenerationError.invalidInput(error.localizedDescription)
        }
    }

    private static func suggestedVideoModelRoot(for variant: LTXVideoVariant) -> String? {
        if let envPath = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_MODEL_ROOT"], !envPath.isEmpty,
           isNativeVideoModelRootAvailable(at: envPath) {
            return envPath
        }

        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let zeroModels = MereRunModelPaths.modelsDir
        let candidates: [String] = {
            switch variant {
            case .distilled:
                return [
                    zeroModels.appendingPathComponent("LTX-2-distilled-bf16", isDirectory: true).path,
                    zeroModels.appendingPathComponent("ltx-video-distilled", isDirectory: true).path,
                    home.appendingPathComponent("models/LTX-2-distilled-bf16", isDirectory: true).path,
                    home.appendingPathComponent("Models/LTX-2-distilled-bf16", isDirectory: true).path,
                ]
            case .unifiedAV:
                return [
                    zeroModels.appendingPathComponent("video-ltx23-full-mlx", isDirectory: true).path,
                    zeroModels.appendingPathComponent("video-ltx23-a2vid-mlx", isDirectory: true).path,
                    zeroModels.appendingPathComponent("video-ltx-av", isDirectory: true).path,
                    zeroModels.appendingPathComponent("video-ltx23-av-mlx", isDirectory: true).path,
                    zeroModels.appendingPathComponent("LTX-2-mlx-av", isDirectory: true).path,
                    home.appendingPathComponent("models/video-ltx-av", isDirectory: true).path,
                    home.appendingPathComponent("models/video-ltx23-full-mlx", isDirectory: true).path,
                    home.appendingPathComponent("models/video-ltx23-a2vid-mlx", isDirectory: true).path,
                    home.appendingPathComponent("models/video-ltx23-av-mlx", isDirectory: true).path,
                    home.appendingPathComponent("models/LTX-2-mlx-av", isDirectory: true).path,
                    home.appendingPathComponent("Models/LTX-2-mlx-av", isDirectory: true).path,
                ]
            }
        }()

        for candidate in candidates where isNativeVideoModelRootAvailable(at: candidate) {
            return candidate
        }
        return nil
    }

    private static func isNativeVideoModelRootAvailable(at path: String) -> Bool {
        let rootURL = URL(fileURLWithPath: path).standardizedFileURL
        do {
            try validate(rootURL)
            return true
        } catch {
            return false
        }
    }
}
