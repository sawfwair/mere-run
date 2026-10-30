import Foundation
import MediaIO
@preconcurrency import MLX
import MLXRandom

public struct IdolRunManifest: Codable, Sendable {
    public let schemaVersion: Int
    public let status: String
    public let backend: String
    public let repository: String
    public let revision: String
    public let referenceRevision: String
    public let inputPath: String
    public let inputSHA256: String
    public let mode: IdolInputMode
    public let alreadyFramed: Bool
    public let settings: IdolSamplingSettings
    public let gaussianCount: Int
    public let camera: IdolCamera
    public let checkpoints: [ModelArtifactPin]
    public let artifacts: [ModelArtifactPin]
}

public enum IdolGenerator {
    /// Model instances are scoped to each stage, releasing encoders before sampling.
    public static func generate(imageURL: URL, outputDirectory: URL, resources: IdolResources,
                                mode: IdolInputMode = .synthetic, alreadyFramed: Bool = false,
                                settings: IdolSamplingSettings, progress: ((String) -> Void)? = nil) throws -> IdolRunManifest {
        let admitted = try VFXImageInputSnapshotBatch.capture([imageURL])
        defer { admitted.cleanup() }
        let image = try MediaImageIO.decode(admitted.snapshotURLs[0])
        _ = try VFXImageInputValidator.validate(width: image.width, height: image.height, path: imageURL.path)
                guard !FileManager.default.fileExists(atPath: outputDirectory.path) else {
            throw IdolError.invalidInput("Output directory already exists: \(outputDirectory.path)")
        }
        progress?("Verifying IDOL checkpoints and recording encoder hashes")
        let checkpoints = try resources.verify(), configuration = try resources.configuration()
        let prepared = try IdolPreprocessor.prepare(image, mode: mode, alreadyFramed: alreadyFramed) { source in
            try IdolBackgroundRemover.remove(source, root: resources.root, progress: progress)
        }
        Memory.clearCache()
        MLXRandom.seed(settings.seed)
        let pixels = MLXArray(MediaImageIO.rgbCHWFloat(prepared, normalizedToMinusOneToOne: false)).reshaped(1, 3, 1024, 1024)
        progress?("Encoding image with native DINOv3 H+")
        let dino = try IdolConditioner.dino(pixels, root: resources.dinoRoot)
        Memory.clearCache()
        progress?("Sampling FLUX.2 VAE posterior")
        let vae = try IdolConditioner.vae(pixels, root: resources.vaeRoot)
        Memory.clearCache()
        let conditioning = IdolConditioning(dino: dino, vae: vae)
        progress?("Loading native IDOL denoiser")
        let state = try sample(configuration: configuration, resources: resources, conditioning: conditioning,
                               settings: settings, progress: progress)
        Memory.clearCache()
        progress?("Sampling octree anchors")
        let anchors = try sampleAnchors(configuration: configuration, resources: resources, state: state,
                                        count: settings.gaussianCount / 32)
        Memory.clearCache()
        progress?("Decoding \(settings.gaussianCount) Gaussian splats")
        let decoder = try IdolDecoder(configuration: configuration, gaussian: true,
            weights: MLX.loadArrays(url: resources.root.appending(path: "decoder_gs.safetensors")))
        let features = try decoder.predict(points: anchors, latent: state.latent)
        let gaussians = try decoder.representation(points: anchors, features: features)
        let camera = try IdolCamera.decode(state.camera.asArray(Float.self))
        progress?("Exporting PLY, prepared input, camera and run manifest")
        let parent = outputDirectory.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appending(path: ".idol-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try gaussians.plyData().write(to: staging.appending(path: "object.ply"))
        try MediaImageIO.writePNG(prepared, to: staging.appending(path: "input.png"))
        try encoder.encode(camera).write(to: staging.appending(path: "camera.json"))
        let artifacts = try ["object.ply", "input.png", "camera.json"].map { name in
            let file = staging.appending(path: name)
            return ModelArtifactPin(filename: name, byteCount: try ModelArtifactPin.fileByteCount(file),
                                    sha256: try ModelArtifactPin.fileSHA256(file))
        }
        let manifest = IdolRunManifest(schemaVersion: 1, status: "completed", backend: "mere.run-native-mlx",
            repository: IdolResources.repository, revision: IdolResources.revision,
            referenceRevision: IdolResources.referenceRevision, inputPath: imageURL.path,
            inputSHA256: try ModelArtifactPin.fileSHA256(admitted.snapshotURLs[0]), mode: mode, alreadyFramed: alreadyFramed,
            settings: settings, gaussianCount: gaussians.count, camera: camera, checkpoints: checkpoints, artifacts: artifacts)
        try encoder.encode(manifest).write(to: staging.appending(path: "manifest.json"))
        try Task.checkCancellation()
        try FileManager.default.moveItem(at: staging, to: outputDirectory)
        return manifest
    }

    private static func sample(configuration: IdolConfiguration, resources: IdolResources, conditioning: IdolConditioning,
                               settings: IdolSamplingSettings, progress: ((String) -> Void)?) throws -> IdolFlowState {
        let model = try IdolDenoiser(configuration: configuration,
            weights: MLX.loadArrays(url: resources.root.appending(path: "denoiser.safetensors")))
        let args = configuration.denoiser.args
        let noise = IdolFlowState(latent: MLXRandom.normal([1, args.q_token_length, args.in_channels]),
                                  camera: MLXRandom.normal([1, 1, args.cam_channels]))
        return try model.sample(noise: noise, conditioning: conditioning, settings: settings) { step in
            progress?("Sampling IDOL flow \(step)/\(settings.steps)")
        }
    }

    private static func sampleAnchors(configuration: IdolConfiguration, resources: IdolResources,
                                      state: IdolFlowState, count: Int) throws -> MLXArray {
        let decoder = try IdolDecoder(configuration: configuration, gaussian: false,
            weights: MLX.loadArrays(url: resources.root.appending(path: "decoder.safetensors")))
        return try decoder.anchors(latent: state.latent, count: count, level: configuration.max_voxel_level)
    }
}
