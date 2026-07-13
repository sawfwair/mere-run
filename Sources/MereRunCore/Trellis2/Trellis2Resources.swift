import Foundation

public struct Trellis2Checkpoint: Hashable, Sendable {
    public let modelID: String
    public let rootURL: URL
    public let sparseStructureFlowURL: URL
    public let sparseStructureDecoderURL: URL
    public let shapeFlowURL: URL
    public let shapeDecoderURL: URL
    public let textureFlowURL: URL
    public let textureDecoderURL: URL
    public let dinoV3URL: URL

    public init(rootURL: URL) throws {
        let root = rootURL.standardizedFileURL
        let validation = Trellis2Resources.validationMessages(in: root)
        guard validation.isEmpty else {
            throw Trellis2ResourceError.invalidCheckpoint(validation.joined(separator: "; "))
        }
        self.init(verifiedRootURL: root)
    }

    /// Tests and already-verified managed-resolution paths can construct the
    /// value without re-hashing eleven gigabytes. This remains module-internal
    /// so public callers cannot bypass checkpoint admission.
    init(verifiedRootURL rootURL: URL) {
        let root = rootURL.standardizedFileURL
        self.modelID = Trellis2Resources.defaultModelID
        self.rootURL = root
        self.sparseStructureFlowURL = root.appendingPathComponent(
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
        )
        self.shapeFlowURL = root.appendingPathComponent(
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
        )
        self.textureFlowURL = root.appendingPathComponent(
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"
        )
        self.shapeDecoderURL = root.appendingPathComponent(
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
        )
        self.textureDecoderURL = root.appendingPathComponent(
            "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"
        )
        self.sparseStructureDecoderURL = root.appendingPathComponent(
            "dependencies/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
        )
        self.dinoV3URL = root.appendingPathComponent("dependencies/dinov3/model.safetensors")
    }
}

public enum Trellis2ResourceError: Error, Equatable, LocalizedError, Sendable {
    case unsupportedModel(String)
    case checkpointNotFound(String)
    case invalidCheckpoint(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let model):
            "Unsupported TRELLIS.2 model '\(model)'. Only \(Trellis2Resources.defaultModelID) is permitted."
        case .checkpointNotFound(let path):
            "TRELLIS.2 checkpoint directory was not found: \(path)"
        case .invalidCheckpoint(let detail):
            "Invalid TRELLIS.2 checkpoint: \(detail)"
        }
    }
}

public enum Trellis2Resources {
    public static let defaultModelID = ModelResolver.ModelID.image3DTrellis2.rawValue
    public static let repository = "microsoft/TRELLIS.2-4B"
    public static let revision = "af44b45f2e35a493886929c6d786e563ec68364d"
    public static let sparseStructureRepository = "microsoft/TRELLIS-image-large"
    public static let sparseStructureRevision = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
    public static let dinoV3Repository = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    public static let dinoV3Revision = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    public static let primaryWeightsSHA256 =
        "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"

    public static let primaryHubFallback = HubFallbackConfig(
        repoId: repository,
        revision: revision,
        patterns: [
            "README.md",
            "pipeline.json",
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
            "ckpts/shape_dec_next_dc_f16c32_fp16.json",
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "ckpts/tex_dec_next_dc_f16c32_fp16.json",
            "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        ]
    )

    public static let mountedHubFallbacks = [
        MountedHubFallbackConfig(
            destinationPath: "dependencies/trellis-image-large",
            hubFallback: HubFallbackConfig(
                repoId: sparseStructureRepository,
                revision: sparseStructureRevision,
                patterns: [
                    "README.md",
                    "ckpts/ss_dec_conv3d_16l8_fp16.json",
                    "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                ]
            )
        ),
        MountedHubFallbackConfig(
            destinationPath: "dependencies/dinov3",
            hubFallback: HubFallbackConfig(
                repoId: dinoV3Repository,
                revision: dinoV3Revision,
                patterns: [
                    "LICENSE.md",
                    "README.md",
                    "config.json",
                    "preprocessor_config.json",
                    "model.safetensors",
                ]
            )
        ),
    ]

    static let artifactPins = [
        ModelArtifactPin(
            filename: "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            byteCount: 2_584_426_920,
            sha256: "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"
        ),
        ModelArtifactPin(
            filename: "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            byteCount: 2_584_574_424,
            sha256: "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
        ),
        ModelArtifactPin(
            filename: "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
            byteCount: 2_584_672_728,
            sha256: "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d"
        ),
        ModelArtifactPin(
            filename: "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            byteCount: 948_490_494,
            sha256: "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581"
        ),
        ModelArtifactPin(
            filename: "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
            byteCount: 948_458_812,
            sha256: "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c"
        ),
        ModelArtifactPin(
            filename: "dependencies/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            byteCount: 147_591_972,
            sha256: "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a"
        ),
        ModelArtifactPin(
            filename: "dependencies/dinov3/model.safetensors",
            byteCount: 1_212_559_808,
            sha256: "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        ),
    ]

    public static let checkpointComponentManifests: [Trellis2CheckpointComponentManifest] = [
        .init(name: "sparse-structure-flow", repository: repository, revision: revision,
              relativePath: artifactPins[0].filename, byteCount: artifactPins[0].byteCount,
              sha256: artifactPins[0].sha256, license: "MIT"),
        .init(name: "shape-flow-512", repository: repository, revision: revision,
              relativePath: artifactPins[1].filename, byteCount: artifactPins[1].byteCount,
              sha256: artifactPins[1].sha256, license: "MIT"),
        .init(name: "texture-flow-512", repository: repository, revision: revision,
              relativePath: artifactPins[2].filename, byteCount: artifactPins[2].byteCount,
              sha256: artifactPins[2].sha256, license: "MIT"),
        .init(name: "shape-decoder", repository: repository, revision: revision,
              relativePath: artifactPins[3].filename, byteCount: artifactPins[3].byteCount,
              sha256: artifactPins[3].sha256, license: "MIT"),
        .init(name: "texture-decoder", repository: repository, revision: revision,
              relativePath: artifactPins[4].filename, byteCount: artifactPins[4].byteCount,
              sha256: artifactPins[4].sha256, license: "MIT"),
        .init(name: "sparse-structure-decoder", repository: sparseStructureRepository,
              revision: sparseStructureRevision, relativePath: artifactPins[5].filename,
              byteCount: artifactPins[5].byteCount, sha256: artifactPins[5].sha256, license: "MIT"),
        .init(name: "dinov3-vitl16", repository: dinoV3Repository, revision: dinoV3Revision,
              relativePath: artifactPins[6].filename, byteCount: artifactPins[6].byteCount,
              sha256: artifactPins[6].sha256, license: "DINOv3 License"),
    ]

    private static let requiredRelativePaths = [
        "README.md",
        "pipeline.json",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
        "ckpts/shape_dec_next_dc_f16c32_fp16.json",
        "ckpts/tex_dec_next_dc_f16c32_fp16.json",
        "dependencies/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json",
        "dependencies/dinov3/LICENSE.md",
        "dependencies/dinov3/config.json",
    ]

    public static func resolve(requestedModel: String?) async throws -> Trellis2Checkpoint {
        let requested = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !requested.isEmpty {
            let explicit = URL(fileURLWithPath: requested).standardizedFileURL
            if FileManager.default.fileExists(atPath: explicit.path) {
                return try Trellis2Checkpoint(rootURL: explicit)
            }
            if requested.contains("/") || requested.hasPrefix(".") || requested.hasPrefix("~") {
                throw Trellis2ResourceError.checkpointNotFound(explicit.path)
            }
        }
        let modelID = requested.isEmpty ? defaultModelID : requested.lowercased()
        guard modelID == defaultModelID else {
            throw Trellis2ResourceError.unsupportedModel(modelID)
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: modelID,
            defaultModelID: defaultModelID,
            allowAutoDownload: false
        )
        // Managed resolution has already run the checksum-backed TRELLIS.2
        // validator. Avoid reading eleven gigabytes a second time at startup.
        return Trellis2Checkpoint(verifiedRootURL: resolution.url)
    }

    public static func validate(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        var invalid = requiredRelativePaths
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        for pin in artifactPins {
            do {
                _ = try pin.verify(in: rootURL, fileManager: fileManager)
            } catch {
                invalid.append(rootURL.appendingPathComponent(pin.filename))
            }
        }
        return invalid
    }

    public static func validationMessages(
        in rootURL: URL,
        fileManager: FileManager = .default
    ) -> [String] {
        validate(rootURL: rootURL, fileManager: fileManager).map {
            "Missing or mismatched TRELLIS.2 artifact: \($0.path)"
        }
    }
}
