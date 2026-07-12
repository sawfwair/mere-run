import Foundation

public struct GeometryModelPin: Codable, Equatable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let sourceCodeRepository: String
    public let sourceCodeRevision: String
    public let license: String
    public let artifacts: [ModelArtifactPin]

    public init(
        modelID: String,
        repository: String,
        revision: String,
        sourceCodeRepository: String,
        sourceCodeRevision: String,
        license: String,
        artifacts: [ModelArtifactPin]
    ) {
        self.modelID = modelID
        self.repository = repository
        self.revision = revision
        self.sourceCodeRepository = sourceCodeRepository
        self.sourceCodeRevision = sourceCodeRevision
        self.license = license
        self.artifacts = artifacts
    }

    @discardableResult
    public func verify(in rootURL: URL, fileManager: FileManager = .default) throws -> [URL] {
        try artifacts.map { try $0.verify(in: rootURL, fileManager: fileManager) }
    }
}

public enum GeometryModelPins {
    public static let moge2Small = GeometryModelPin(
        modelID: ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue,
        repository: "Ruicheng/moge-2-vits-normal-onnx",
        revision: "e50ffda41565591092adea54c6ac83d6212e1e23",
        sourceCodeRepository: "microsoft/MoGe",
        sourceCodeRevision: "07444410f1e33f402353b99d6ccd26bd31e469e8",
        license: "MIT",
        artifacts: [
            ModelArtifactPin(
                filename: "model.onnx",
                byteCount: 140_852_051,
                sha256: "24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0"
            ),
        ]
    )

    public static let videoDepthAnythingSmall = GeometryModelPin(
        modelID: ModelResolver.ModelID.visionDepthVDASmall.rawValue,
        repository: "depth-anything/Video-Depth-Anything-Small",
        revision: "256875362cff76724b920335dfb4b29dd611f66e",
        sourceCodeRepository: "DepthAnything/Video-Depth-Anything",
        sourceCodeRevision: "4f5ae23172ba60fd7bc11ef671cca678842c7072",
        license: "Apache-2.0",
        artifacts: [
            ModelArtifactPin(
                filename: "video_depth_anything_vits.pth",
                byteCount: 116_440_756,
                sha256: "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609"
            ),
        ]
    )

    public static let videoDepthAnythingSmallMetric = GeometryModelPin(
        modelID: ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
        repository: "depth-anything/Metric-Video-Depth-Anything-Small",
        revision: "273d090f2ce17df50c2872d82c8322c45da5b4dd",
        sourceCodeRepository: "DepthAnything/Video-Depth-Anything",
        sourceCodeRevision: "4f5ae23172ba60fd7bc11ef671cca678842c7072",
        license: "Apache-2.0",
        artifacts: [
            ModelArtifactPin(
                filename: "metric_video_depth_anything_vits.pth",
                byteCount: 116_444_063,
                sha256: "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"
            ),
        ]
    )

    public static let depthAnything3Small = GeometryModelPin(
        modelID: ModelResolver.ModelID.visionGeometryDA3Small.rawValue,
        repository: "depth-anything/DA3-SMALL",
        revision: "e08cab65ca0ec38e7826075418411ab90cab4da3",
        sourceCodeRepository: "ByteDance-Seed/Depth-Anything-3",
        sourceCodeRevision: "41736238f5bced4debf3f2a12375d2466874866d",
        license: "Apache-2.0",
        artifacts: [
            ModelArtifactPin(
                filename: "config.json",
                byteCount: 1_202,
                sha256: "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2"
            ),
            ModelArtifactPin(
                filename: "model.safetensors",
                byteCount: 137_248_940,
                sha256: "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
            ),
        ]
    )

    public static let tripoSR = GeometryModelPin(
        modelID: ModelResolver.ModelID.image3DTripoSR.rawValue,
        repository: "stabilityai/TripoSR",
        revision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
        sourceCodeRepository: "VAST-AI-Research/TripoSR",
        sourceCodeRevision: "107cefdc244c39106fa830359024f6a2f1c78871",
        license: "MIT",
        artifacts: [
            ModelArtifactPin(
                filename: "config.yaml",
                byteCount: 987,
                sha256: "74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62"
            ),
            ModelArtifactPin(
                filename: "model.ckpt",
                byteCount: 1_677_246_742,
                sha256: "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
            ),
        ]
    )

    /// Reconstruction-only package. The CC-BY-NC Zero123++ derivative is
    /// intentionally excluded; callers must supply licensed 4/6-view input.
    public static let instantMeshBase = GeometryModelPin(
        modelID: ModelResolver.ModelID.image3DInstantMeshBase.rawValue,
        repository: "TencentARC/InstantMesh",
        revision: "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
        sourceCodeRepository: "TencentARC/InstantMesh",
        sourceCodeRevision: "08822c52fdc399b93ea00e4fa9e596344ed52ccc",
        license: "Apache-2.0 reconstruction weights; view generation excluded",
        artifacts: [
            ModelArtifactPin(
                filename: "instant_mesh_base.ckpt",
                byteCount: 1_253_574_354,
                sha256: "22701cd25201d624ebb1568b93cf91b43a2c32006835c08fe73e1f3c9f6c44b5"
            ),
        ]
    )

    public static let all: [GeometryModelPin] = [
        moge2Small,
        videoDepthAnythingSmall,
        videoDepthAnythingSmallMetric,
        depthAnything3Small,
        tripoSR,
        instantMeshBase,
    ]

    public static func pin(for modelID: String) -> GeometryModelPin? {
        all.first { $0.modelID == modelID }
    }
}
