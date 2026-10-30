import Foundation
import MediaIO
import MLX
import XCTest
@testable import MereRunCore

/// Opt-in real-checkpoint parity. The normal gate uses committed miniature oracles.
final class IdolQualificationTests: MereRunCoreTestCase {
    func testReleasedBiRefNetAgainstTorchvisionFloat32() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_TEST_IDOL_ROOT"] else {
            throw XCTSkip("Set MERERUN_TEST_IDOL_ROOT for real BiRefNet checkpoint parity.")
        }
        let root = URL(fileURLWithPath: path)
        let trace = try MLX.loadArrays(url: root.appending(path: "reference-background/trace.safetensors"))
        let model = try BiRefNet(weights: MLX.loadArrays(url: root.appending(path: "checkpoint/birefnet.safetensors")))
        let actual = try model(trace["biref_input"]!.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2).asType(.float32)
        let expected = trace["biref_alpha"]!.asType(.float32)
        let difference = actual - expected
        let rmse = sqrt(mean(difference * difference)).item(Float.self)
        let mask = (actual .> 0.5).asType(.float32), reference = (expected .> 0.5).asType(.float32)
        let intersection = sum(mask * reference).item(Float.self)
        let union = sum(maximum(mask, reference)).item(Float.self)
        let iou = intersection / union
        print("IDOL real BiRefNet: mask RMSE=\(rmse), IoU=\(iou)")
        XCTAssertLessThanOrEqual(rmse, 0.005)
        XCTAssertGreaterThanOrEqual(iou, 0.99)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(["mask_rmse": rmse, "foreground_iou": iou]).write(to: root.appending(path: "native-background-parity.json"))
    }

    func testReleasedCheckpointAgainstUpstreamTrace() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_TEST_IDOL_ROOT"] else {
            throw XCTSkip("Set MERERUN_TEST_IDOL_ROOT to a local checkpoint qualification directory.")
        }
        let root = URL(fileURLWithPath: path), checkpoint = root.appending(path: "checkpoint")
        let reference = root.appending(path: "reference-submarine")
        let trace = try MLX.loadArrays(url: reference.appending(path: "trace.safetensors"))
        let configuration = try JSONDecoder().decode(IdolConfiguration.self, from: Data(contentsOf: checkpoint.appending(path: "config.json")))
        let image = try MediaImageIO.decode(reference.appending(path: "input.png"))
        let pixels = MLXArray(MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: false)).reshaped(1, 3, 1024, 1024)
        var metrics: [String: Float] = [:]
        func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray, relative: Float) {
            XCTAssertEqual(actual.shape, expected.shape)
            let difference = actual.asType(.float32) - expected.asType(.float32)
            let error = sqrt(mean(difference * difference)).item(Float.self)
            let signal = sqrt(mean(expected.asType(.float32) * expected.asType(.float32))).item(Float.self)
            metrics[name + "_rmse"] = error; metrics[name + "_relative_rmse"] = error / max(signal, 1e-8)
            print("IDOL qualification \(name): RMSE=\(error), relative=\(error / max(signal, 1e-8))")
            XCTAssertLessThanOrEqual(error / max(signal, 1e-8), relative, name)
        }
        let dino = try IdolConditioner.dino(pixels, root: root.appending(path: "dinov3"))
        compare("dino", dino, trace["feature1"]!, relative: 0.025)
        Memory.clearCache()
        let vae = try IdolConditioner.vae(pixels, root: root.appending(path: "vae"), posteriorNoise: trace["vae_noise"]!)
        compare("vae", vae, trace["feature2"]!, relative: 0.04)
        Memory.clearCache()
        // Fixed upstream conditioning and noise isolate transformer arithmetic from RNG and encoder differences.
        let condition = IdolConditioning(dino: trace["feature1"]!, vae: trace["feature2"]!)
        let model = try IdolDenoiser(configuration: configuration, weights: MLX.loadArrays(url: checkpoint.appending(path: "denoiser.safetensors")))
        let noise = IdolFlowState(latent: trace["noise_latent"]!, camera: trace["noise_camera"]!)
        let positive = try model.velocity(noise, time: 1000, conditioning: condition)
        compare("positive_latent", positive.latent, trace["positive_latent"]!, relative: 0.01)
        compare("positive_camera", positive.camera, trace["positive_camera"]!, relative: 0.01)
        let negative = try model.velocity(noise, time: 1000, conditioning: condition.zeros)
        compare("negative_latent", negative.latent, trace["negative_latent"]!, relative: 0.01)
        let sampled = try model.sample(noise: noise, conditioning: condition, settings: IdolSamplingSettings())
        compare("flow_latent", sampled.latent, trace["flow_latent"]!, relative: 0.03)
        compare("flow_camera", sampled.camera, trace["flow_camera"]!, relative: 0.03)
        let octree = try IdolDecoder(configuration: configuration, gaussian: false,
            weights: MLX.loadArrays(url: checkpoint.appending(path: "decoder.safetensors")))
        for resolution in [2, 16, 256] {
            let logits = try octree.predict(points: trace["octree_queries"]!, latent: trace["flow_latent"]!, resolution: resolution)
            compare("octree_\(resolution)", logits, trace["octree_\(resolution)"]!, relative: 0.01)
        }
        let decoder = try IdolDecoder(configuration: configuration, gaussian: true,
            weights: MLX.loadArrays(url: checkpoint.appending(path: "decoder_gs.safetensors")))
        let features = try decoder.predict(points: trace["anchors"]!, latent: trace["flow_latent"]!)
        compare("gaussian_features", features, trace["gaussian_features"]!, relative: 0.01)
        let gaussian = try decoder.representation(points: trace["anchors"]!, features: features)
        try gaussian.plyData().write(to: root.appending(path: "native-matched.ply"))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metrics).write(to: root.appending(path: "native-parity.json"))
    }
}
