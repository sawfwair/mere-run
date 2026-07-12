import Foundation

public struct GeometryModelLicenseEvidence: Codable, Equatable, Sendable {
    public let resourceName: String
    public let resourcePin: ModelArtifactPin
    public let installedPin: ModelArtifactPin

    public init(
        resourceName: String,
        byteCount: Int64,
        sha256: String,
        resourceByteCount: Int64? = nil,
        resourceSHA256: String? = nil
    ) {
        self.resourceName = resourceName
        self.resourcePin = ModelArtifactPin(
            filename: resourceName,
            byteCount: resourceByteCount ?? byteCount,
            sha256: resourceSHA256 ?? sha256
        )
        self.installedPin = ModelArtifactPin(
            filename: "MERERUN_UPSTREAM_LICENSE",
            byteCount: byteCount,
            sha256: sha256
        )
    }
}

public enum GeometryModelLicenseError: Error, Equatable, LocalizedError, Sendable {
    case bundledEvidenceMissing(String)

    public var errorDescription: String? {
        switch self {
        case .bundledEvidenceMissing(let name):
            "Bundled upstream VFX license evidence is missing: \(name)."
        }
    }
}

public struct GeometryModelPin: Codable, Equatable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let sourceCodeRepository: String
    public let sourceCodeRevision: String
    public let license: String
    public let licenseEvidence: GeometryModelLicenseEvidence?
    public let artifacts: [ModelArtifactPin]

    public init(
        modelID: String,
        repository: String,
        revision: String,
        sourceCodeRepository: String,
        sourceCodeRevision: String,
        license: String,
        licenseEvidence: GeometryModelLicenseEvidence? = nil,
        artifacts: [ModelArtifactPin]
    ) {
        self.modelID = modelID
        self.repository = repository
        self.revision = revision
        self.sourceCodeRepository = sourceCodeRepository
        self.sourceCodeRevision = sourceCodeRevision
        self.license = license
        self.licenseEvidence = licenseEvidence
        self.artifacts = artifacts
    }

    public var runtimeArtifacts: [ModelArtifactPin] {
        artifacts + (licenseEvidence.map { [$0.installedPin] } ?? [])
    }

    @discardableResult
    public func verify(in rootURL: URL, fileManager: FileManager = .default) throws -> [URL] {
        try runtimeArtifacts.map { try $0.verify(in: rootURL, fileManager: fileManager) }
    }

    public func installBundledLicenseEvidence(
        in rootURL: URL,
        fileManager: FileManager = .default
    ) throws {
        guard let licenseEvidence else { return }
        let resourceURL = Bundle.module.url(
            forResource: licenseEvidence.resourceName,
            withExtension: nil,
            subdirectory: "VFXLicenses"
        ) ?? Bundle.module.url(forResource: licenseEvidence.resourceName, withExtension: nil)
        guard let resourceURL else {
            throw GeometryModelLicenseError.bundledEvidenceMissing(licenseEvidence.resourceName)
        }
        let bundlePin = ModelArtifactPin(
            filename: resourceURL.lastPathComponent,
            byteCount: licenseEvidence.resourcePin.byteCount,
            sha256: licenseEvidence.resourcePin.sha256
        )
        _ = try bundlePin.verify(in: resourceURL.deletingLastPathComponent(), fileManager: fileManager)
        let destination = rootURL.appendingPathComponent(licenseEvidence.installedPin.filename)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        if licenseEvidence.resourcePin.byteCount == licenseEvidence.installedPin.byteCount,
           licenseEvidence.resourcePin.sha256 == licenseEvidence.installedPin.sha256 {
            try fileManager.copyItem(at: resourceURL, to: destination)
        } else {
            let resourceBytes = try Data(contentsOf: resourceURL)
            guard licenseEvidence.installedPin.byteCount <= Int64(resourceBytes.count) else {
                throw GeometryModelLicenseError.bundledEvidenceMissing(licenseEvidence.resourceName)
            }
            try Data(resourceBytes.prefix(Int(licenseEvidence.installedPin.byteCount)))
                .write(to: destination, options: .atomic)
        }
        _ = try licenseEvidence.installedPin.verify(in: rootURL, fileManager: fileManager)
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
        licenseEvidence: GeometryModelLicenseEvidence(
            resourceName: "MoGe-LICENSE",
            byteCount: 12_500,
            sha256: "ad7d951c80c5fc2b2bce035f2041bc0a0dbf9028c8ecc4c9a8e1fba8130b6b59"
        ),
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
        licenseEvidence: GeometryModelLicenseEvidence(
            resourceName: "Video-Depth-Anything-LICENSE",
            byteCount: 11_356,
            sha256: "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
            resourceByteCount: 11_357,
            resourceSHA256: "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
        ),
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
        licenseEvidence: GeometryModelLicenseEvidence(
            resourceName: "Video-Depth-Anything-LICENSE",
            byteCount: 11_356,
            sha256: "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
            resourceByteCount: 11_357,
            resourceSHA256: "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
        ),
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
        licenseEvidence: GeometryModelLicenseEvidence(
            resourceName: "Depth-Anything-3-LICENSE",
            byteCount: 11_355,
            sha256: "78446e29c48900cda82620a8df183cca61f0a595e05a49d0401a5fd604dd1870"
        ),
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
        licenseEvidence: GeometryModelLicenseEvidence(
            resourceName: "TripoSR-LICENSE",
            byteCount: 1_080,
            sha256: "ade0a66629bdd7e01e46b3296b3851cff0fd27989bca53da470ad6e96ed620fb"
        ),
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
