import Foundation
import MediaIO
@preconcurrency import MLX
import MLXRandom

public enum Trellis2Progress: Equatable, Sendable {
    case verifyingCheckpoint
    case decodingImage
    case preprocessingImage
    case encodingImage
    case samplingSparseStructure(completed: Int, total: Int)
    case decodingSparseStructure
    case samplingShape(completed: Int, total: Int)
    case samplingTexture(completed: Int, total: Int)
    case decodingShape
    case decodingTexture
    case extractingMesh
    case remeshingMesh
    case exportingAssets

    public var message: String {
        switch self {
        case .verifyingCheckpoint:
            "Verifying pinned TRELLIS.2 and DINOv3 checkpoints"
        case .decodingImage:
            "Decoding source image"
        case .preprocessingImage:
            "Preparing 512px DINOv3 conditioning image"
        case .encodingImage:
            "Encoding image with native MLX DINOv3 ViT-L/16"
        case .samplingSparseStructure(let completed, let total):
            "Sampling sparse structure \(completed)/\(total)"
        case .decodingSparseStructure:
            "Decoding occupied 32-grid coordinates"
        case .samplingShape(let completed, let total):
            "Sampling O-Voxel shape latent \(completed)/\(total)"
        case .samplingTexture(let completed, let total):
            "Sampling PBR texture latent \(completed)/\(total)"
        case .decodingShape:
            "Decoding shape and adaptive subdivision tree"
        case .decodingTexture:
            "Decoding six-channel PBR O-Voxels"
        case .extractingMesh:
            "Extracting flexible-dual-grid mesh and filling small boundary holes"
        case .remeshingMesh:
            "Remeshing to the watertight narrow-band dual-contour envelope"
        case .exportingAssets:
            "Writing OBJ, PLY, GLB, PBR voxels, and manifests"
        }
    }
}

public enum Trellis2GeneratorError: Error, Equatable, LocalizedError, Sendable {
    case inputImageNotFound(String)
    case invalidMaximumSparseTokens(Int)

    public var errorDescription: String? {
        switch self {
        case .inputImageNotFound(let path):
            "TRELLIS.2 input image was not found: \(path)"
        case .invalidMaximumSparseTokens(let value):
            "TRELLIS.2 maximum sparse tokens must be at least 4096; received \(value)."
        }
    }
}

public struct Trellis2RunResult: Sendable {
    public let meshExport: MeshExportResult
    public let pbrExport: Trellis2PBRVoxelExport
    public let runManifest: Trellis2RunManifestExport
    public let checkpoint: Trellis2Checkpoint
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let croppedTransparentForeground: Bool
    public let foregroundPolicy: String
    public let seed: UInt64
    public let maximumSparseTokens: Int

    public init(
        meshExport: MeshExportResult,
        pbrExport: Trellis2PBRVoxelExport,
        runManifest: Trellis2RunManifestExport,
        checkpoint: Trellis2Checkpoint,
        sourceWidth: Int,
        sourceHeight: Int,
        croppedTransparentForeground: Bool,
        foregroundPolicy: String,
        seed: UInt64,
        maximumSparseTokens: Int
    ) {
        self.meshExport = meshExport
        self.pbrExport = pbrExport
        self.runManifest = runManifest
        self.checkpoint = checkpoint
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.croppedTransparentForeground = croppedTransparentForeground
        self.foregroundPolicy = foregroundPolicy
        self.seed = seed
        self.maximumSparseTokens = maximumSparseTokens
    }
}

public actor Trellis2Generator {
    /// Safety ceiling for decoded 512-resolution O-Voxels. Microsoft's
    /// 49,152-token option is only used to select a cascade resolution and is
    /// not a decoder limit for the direct 512 pipeline.
    public static let defaultMaximumSparseTokens = 2_097_152

    public init() {}

    public func generate(
        imageURL: URL,
        outputDirectory: URL,
        model requestedModel: String? = nil,
        foregroundPolicy: Trellis2ForegroundPolicy = .transparentAlpha,
        seed: UInt64 = 42,
        maximumSparseTokens: Int = defaultMaximumSparseTokens,
        remesh: Trellis2RemeshConfiguration? = Trellis2RemeshConfiguration(),
        progress: (@Sendable (Trellis2Progress) -> Void)? = nil
    ) async throws -> Trellis2RunResult {
        let inputURL = imageURL.standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw Trellis2GeneratorError.inputImageNotFound(inputURL.path)
        }
        guard maximumSparseTokens >= 4_096 else {
            throw Trellis2GeneratorError.invalidMaximumSparseTokens(maximumSparseTokens)
        }

        let admittedInput = try VFXImageInputSnapshotBatch.capture([inputURL])
        defer { admittedInput.cleanup() }
        let snapshotURL = admittedInput.snapshotURLs[0]

        progress?(.verifyingCheckpoint)
        let checkpoint = try await Trellis2Resources.resolve(requestedModel: requestedModel)

        progress?(.decodingImage)
        let image = try MediaImageIO.decode(snapshotURL)
        _ = try VFXImageInputValidator.validate(
            width: image.width,
            height: image.height,
            path: inputURL.path
        )

        progress?(.preprocessingImage)
        let prepared = try Trellis2Preprocessor.prepare(
            image: image,
            size: 512,
            foregroundPolicy: foregroundPolicy
        )

        progress?(.encodingImage)
        let condition = try encodeCondition(
            prepared.conditionInput,
            checkpointURL: checkpoint.dinoV3URL
        )
        let negativeCondition = MLX.zeros(condition.shape, dtype: condition.dtype)
        MLX.eval(condition, negativeCondition)

        MLXRandom.seed(seed)
        let structureCoordinates = Self.denseCoordinates(resolution: 16)
        let structureLatent = try sampleStructure(
            coordinates: structureCoordinates,
            condition: condition,
            negativeCondition: negativeCondition,
            checkpointURL: checkpoint.sparseStructureFlowURL,
            progress: progress
        )

        progress?(.decodingSparseStructure)
        let coordinates = try decodeStructure(
            structureLatent,
            checkpointURL: checkpoint.sparseStructureDecoderURL,
            maximumSparseTokens: maximumSparseTokens
        )

        let shapeLatents = try sampleShape(
            coordinates: coordinates,
            condition: condition,
            negativeCondition: negativeCondition,
            checkpointURL: checkpoint.shapeFlowURL,
            progress: progress
        )
        let textureLatent = try sampleTexture(
            coordinates: coordinates,
            normalizedShape: shapeLatents.normalized,
            condition: condition,
            negativeCondition: negativeCondition,
            checkpointURL: checkpoint.textureFlowURL,
            progress: progress
        )

        progress?(.decodingShape)
        let shape = try decodeShape(
            latent: shapeLatents.denormalized,
            coordinates: coordinates,
            checkpointURL: checkpoint.shapeDecoderURL,
            maximumSparseTokens: maximumSparseTokens
        )
        progress?(.decodingTexture)
        let texture = try decodeTexture(
            latent: textureLatent,
            coordinates: coordinates,
            guides: shape.subdivisions,
            checkpointURL: checkpoint.textureDecoderURL,
            maximumSparseTokens: maximumSparseTokens
        )

        progress?(.extractingMesh)
        let decoded = try Trellis2FlexibleDualGrid.decode(
            shape: shape.tensor,
            texture: texture,
            resolution: 512
        )
        MLX.Memory.clearCache()

        if remesh != nil {
            progress?(.remeshingMesh)
        }
        progress?(.exportingAssets)
        let policyName = foregroundPolicy == .alreadyFramed
            ? "already-framed"
            : "transparent-alpha"
        let artifacts = try Trellis2ArtifactExporter.export(
            asset: decoded,
            checkpoint: checkpoint,
            inputURL: inputURL,
            inputRecord: admittedInput.inputRecords[0],
            outputDirectory: outputDirectory.standardizedFileURL,
            stem: inputURL.deletingPathExtension().lastPathComponent,
            sourceWidth: prepared.sourceWidth,
            sourceHeight: prepared.sourceHeight,
            foregroundPolicy: policyName,
            croppedTransparentForeground: prepared.croppedTransparentForeground,
            seed: seed,
            maximumSparseTokens: maximumSparseTokens,
            remesh: remesh
        )
        MLX.Memory.clearCache()
        return Trellis2RunResult(
            meshExport: artifacts.mesh,
            pbrExport: artifacts.pbr,
            runManifest: artifacts.run,
            checkpoint: checkpoint,
            sourceWidth: prepared.sourceWidth,
            sourceHeight: prepared.sourceHeight,
            croppedTransparentForeground: prepared.croppedTransparentForeground,
            foregroundPolicy: policyName,
            seed: seed,
            maximumSparseTokens: maximumSparseTokens
        )
    }

    public func unload() {
        MLX.Memory.clearCache()
    }

    private func encodeCondition(_ input: MLXArray, checkpointURL: URL) throws -> MLXArray {
        let condition: MLXArray = try {
            let conditioner = try Trellis2DINOv3Conditioner(checkpointURL: checkpointURL)
            return conditioner.encode(input)
        }()
        MLX.eval(condition)
        MLX.Memory.clearCache()
        return condition
    }

    private func sampleStructure(
        coordinates: [Trellis2VoxelCoordinate],
        condition: MLXArray,
        negativeCondition: MLXArray,
        checkpointURL: URL,
        progress: (@Sendable (Trellis2Progress) -> Void)?
    ) throws -> MLXArray {
        let latent: MLXArray = try {
            let model = try Trellis2FlowModel(
                configuration: .sparseStructure512,
                checkpointURL: checkpointURL
            )
            let noise = MLXRandom.normal([1, coordinates.count, 8]).asType(.float32)
            return try Trellis2FlowSampler.sample(
                noise: noise,
                coordinates: coordinates,
                condition: condition,
                negativeCondition: negativeCondition,
                model: model,
                configuration: .sparseStructure,
                progress: { completed, total in
                    progress?(.samplingSparseStructure(completed: completed, total: total))
                }
            )
        }()
        MLX.eval(latent)
        MLX.Memory.clearCache()
        return latent
    }

    private func decodeStructure(
        _ latent: MLXArray,
        checkpointURL: URL,
        maximumSparseTokens: Int
    ) throws -> [Trellis2VoxelCoordinate] {
        let coordinates: [Trellis2VoxelCoordinate] = try {
            let decoder = try Trellis2StructureDecoder(checkpointURL: checkpointURL)
            return try decoder.decode(
                latentTokens: latent,
                outputResolution: 32,
                maximumTokens: maximumSparseTokens
            )
        }()
        MLX.Memory.clearCache()
        return coordinates
    }

    private func sampleShape(
        coordinates: [Trellis2VoxelCoordinate],
        condition: MLXArray,
        negativeCondition: MLXArray,
        checkpointURL: URL,
        progress: (@Sendable (Trellis2Progress) -> Void)?
    ) throws -> (normalized: MLXArray, denormalized: MLXArray) {
        let normalized: MLXArray = try {
            let model = try Trellis2FlowModel(configuration: .shape512, checkpointURL: checkpointURL)
            let noise = MLXRandom.normal([1, coordinates.count, 32]).asType(.float32)
            return try Trellis2FlowSampler.sample(
                noise: noise,
                coordinates: coordinates,
                condition: condition,
                negativeCondition: negativeCondition,
                model: model,
                configuration: .shape,
                progress: { completed, total in
                    progress?(.samplingShape(completed: completed, total: total))
                }
            )
        }()
        MLX.eval(normalized)
        let denormalized = Self.denormalize(
            normalized,
            mean: Trellis2Normalization.shapeMean,
            standardDeviation: Trellis2Normalization.shapeStandardDeviation
        )
        MLX.eval(denormalized)
        MLX.Memory.clearCache()
        return (normalized, denormalized)
    }

    private func sampleTexture(
        coordinates: [Trellis2VoxelCoordinate],
        normalizedShape: MLXArray,
        condition: MLXArray,
        negativeCondition: MLXArray,
        checkpointURL: URL,
        progress: (@Sendable (Trellis2Progress) -> Void)?
    ) throws -> MLXArray {
        let normalized: MLXArray = try {
            let model = try Trellis2FlowModel(configuration: .texture512, checkpointURL: checkpointURL)
            let noise = MLXRandom.normal([1, coordinates.count, 32]).asType(.float32)
            return try Trellis2FlowSampler.sample(
                noise: noise,
                coordinates: coordinates,
                condition: condition,
                negativeCondition: negativeCondition,
                model: model,
                configuration: .texture,
                staticInput: normalizedShape,
                progress: { completed, total in
                    progress?(.samplingTexture(completed: completed, total: total))
                }
            )
        }()
        MLX.eval(normalized)
        let denormalized = Self.denormalize(
            normalized,
            mean: Trellis2Normalization.textureMean,
            standardDeviation: Trellis2Normalization.textureStandardDeviation
        )
        MLX.eval(denormalized)
        MLX.Memory.clearCache()
        return denormalized
    }

    private func decodeShape(
        latent: MLXArray,
        coordinates: [Trellis2VoxelCoordinate],
        checkpointURL: URL,
        maximumSparseTokens: Int
    ) throws -> Trellis2SparseDecoderOutput {
        let output: Trellis2SparseDecoderOutput = try {
            let decoder = try Trellis2SparseDecoder(kind: .shape, checkpointURL: checkpointURL)
            let sparse = try Trellis2SparseTensor(
                features: latent.squeezed(axis: 0),
                coordinates: coordinates
            )
            return try decoder.decode(sparse, maximumTokens: maximumSparseTokens)
        }()
        MLX.eval(output.tensor.features)
        MLX.Memory.clearCache()
        return output
    }

    private func decodeTexture(
        latent: MLXArray,
        coordinates: [Trellis2VoxelCoordinate],
        guides: [Trellis2Subdivision],
        checkpointURL: URL,
        maximumSparseTokens: Int
    ) throws -> Trellis2SparseTensor {
        let output: Trellis2SparseTensor = try {
            let decoder = try Trellis2SparseDecoder(kind: .texture, checkpointURL: checkpointURL)
            let sparse = try Trellis2SparseTensor(
                features: latent.squeezed(axis: 0),
                coordinates: coordinates
            )
            return try decoder.decode(
                sparse,
                guideSubdivisions: guides,
                maximumTokens: maximumSparseTokens
            ).tensor
        }()
        MLX.eval(output.features)
        MLX.Memory.clearCache()
        return output
    }

    static func denseCoordinates(resolution: Int) -> [Trellis2VoxelCoordinate] {
        var result = [Trellis2VoxelCoordinate]()
        result.reserveCapacity(resolution * resolution * resolution)
        for x in 0..<resolution {
            for y in 0..<resolution {
                for z in 0..<resolution {
                    result.append(Trellis2VoxelCoordinate(x: Int32(x), y: Int32(y), z: Int32(z)))
                }
            }
        }
        return result
    }

    private static func denormalize(
        _ input: MLXArray,
        mean: [Float],
        standardDeviation: [Float]
    ) -> MLXArray {
        let meanArray = MLXArray(mean).reshaped(1, 1, mean.count)
        let stdArray = MLXArray(standardDeviation).reshaped(1, 1, standardDeviation.count)
        return input * stdArray + meanArray
    }
}
