import Foundation

// MARK: - Repository Definitions

public enum ZImageI2LRepository {
    /// Canonical local model id used by the mere.run package and CLI.
    public static let modelId = "i2l"
    /// Backward-compatible alias used by older local installs.
    public static let legacyModelId = "zeta-i2l"

    public static let id = "DiffSynth-Studio/Z-Image-i2L"
    public static let revision = "main"
    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: id,
        revision: revision,
        patterns: ["model.safetensors"]
    )
}

public enum GeneralImageEncodersRepository {
    public static let id = "DiffSynth-Studio/General-Image-Encoders"
    public static let revision = "main"
    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: id,
        revision: revision,
        patterns: [
            "SigLIP2-G384/model.safetensors",
            "DINOv3-7B/model.safetensors",
        ]
    )
}

public enum ZImageRepository {
    public static let id = "Tongyi-MAI/Z-Image"
    public static let revision = "main"
}

// MARK: - Resource Paths

public struct ZImageI2LResources: Sendable, Hashable {
    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    /// Z-Image-i2L model weights candidate locations.
    public var i2lModelCandidates: [URL] {
        [
            rootURL.appendingPathComponent("z-image-i2l/model.safetensors"),
            rootURL.appendingPathComponent("model.safetensors"),
        ]
    }

    /// SigLIP2-G384 encoder weights candidate locations.
    public var siglip2WeightsCandidates: [URL] {
        [
            rootURL.appendingPathComponent("SigLIP2-G384/model.safetensors"),
            rootURL.appendingPathComponent("general-image-encoders/SigLIP2-G384/model.safetensors"),
        ]
    }

    /// DINOv3-7B encoder weights candidate locations.
    public var dinov3WeightsCandidates: [URL] {
        [
            rootURL.appendingPathComponent("DINOv3-7B/model.safetensors"),
            rootURL.appendingPathComponent("general-image-encoders/DINOv3-7B/model.safetensors"),
        ]
    }

    // Z-Image transformer (for applying LoRA)
    public var zImageTransformerURL: URL {
        rootURL.appendingPathComponent("z-image/transformer")
    }

    // Shared components from Z-Image-Turbo
    public var textEncoderURL: URL {
        rootURL.appendingPathComponent("z-image-turbo/text_encoder")
    }

    public var vaeURL: URL {
        rootURL.appendingPathComponent("z-image-turbo/vae")
    }

    public var tokenizerURL: URL {
        rootURL.appendingPathComponent("z-image-turbo/tokenizer")
    }

    public func resolvedI2LModelURL(fileManager: FileManager = .default) -> URL? {
        firstExistingURL(i2lModelCandidates, fileManager: fileManager)
    }

    public func resolvedSigLIP2WeightsURL(fileManager: FileManager = .default) -> URL? {
        firstExistingURL(siglip2WeightsCandidates, fileManager: fileManager)
    }

    public func resolvedDINOv3WeightsURL(fileManager: FileManager = .default) -> URL? {
        firstExistingURL(dinov3WeightsCandidates, fileManager: fileManager)
    }

    public static func resolveNestedIfNeeded(
        base: URL,
        fileManager: FileManager = .default
    ) -> URL {
        let nested = base.appendingPathComponent("i2l", isDirectory: true)
        let nestedResources = ZImageI2LResources(rootURL: nested)
        if nestedResources.validate(fileManager: fileManager).isEmpty {
            return nested
        }
        return base
    }

    public static func resolveInstalledRoot(
        fileManager: FileManager = .default
    ) -> URL? {
        let storageIDs = [
            ZImageI2LRepository.modelId,
            ZImageI2LRepository.legacyModelId,
        ]
        for storageID in storageIDs {
            let base = MereRunModelPaths.resolveModelDir(storageID) { root in
                let resolved = resolveNestedIfNeeded(base: root, fileManager: fileManager)
                return ZImageI2LResources(rootURL: resolved).validate(fileManager: fileManager).isEmpty
            }
            let resolved = resolveNestedIfNeeded(base: base, fileManager: fileManager)
            if ZImageI2LResources(rootURL: resolved).validate(fileManager: fileManager).isEmpty {
                return resolved
            }
        }
        return nil
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if firstExistingURL(i2lModelCandidates, fileManager: fileManager) == nil,
           let expected = i2lModelCandidates.first {
            missing.append(expected)
        }
        if firstExistingURL(siglip2WeightsCandidates, fileManager: fileManager) == nil,
           let expected = siglip2WeightsCandidates.first {
            missing.append(expected)
        }
        if firstExistingURL(dinov3WeightsCandidates, fileManager: fileManager) == nil,
           let expected = dinov3WeightsCandidates.first {
            missing.append(expected)
        }
        return missing
    }

    private func firstExistingURL(_ candidates: [URL], fileManager: FileManager) -> URL? {
        candidates.first { fileManager.fileExists(atPath: $0.path) }
    }
}
