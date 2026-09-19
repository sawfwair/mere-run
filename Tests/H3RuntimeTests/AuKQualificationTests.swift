import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunAudioModels

/// Opt-in trained-checkpoint parity. Generate receipts with qualify-auk-reference.py.
final class AuKQualificationTests: MLXTestCase {
    private struct Metric: Encodable {
        let shape: [Int]
        let relativeL2: Float
        let maximumAbsolute: Float
    }

    func testOriginalCheckpointComponentsExecute() throws {
        guard let path = ProcessInfo.processInfo.environment["AUK_MODEL_ROOT"] else {
            throw XCTSkip("Set AUK_MODEL_ROOT for original checkpoint execution")
        }
        let root = URL(fileURLWithPath: path)
        let variant: AuKVariant = ProcessInfo.processInfo.environment["AUK_VARIANT"] == "flash" ? .flash : .base
        let waveform = sin(MLXArray(0..<24000).asType(.float32) * (2 * Float.pi * 440 / 24000))
            .reshaped(1, 24000, 1) * 0.1
        let reference: MLXArray
        do {
            let vae = try AuKVAE(weights: AuKCheckpoint.vae(root.appendingPathComponent("vae.safetensors")))
            reference = try vae.encode(waveform)
            let decoded = try vae.decode(reference)
            XCTAssertEqual(reference.shape, [1, 50, 64])
            XCTAssertEqual(decoded.shape, waveform.shape)
            XCTAssertTrue(all(isFinite(decoded)).item(Bool.self))
            eval(reference)
        }
        Memory.clearCache()
        let (weights, fusion) = try AuKCheckpoint.diffusion(root.appendingPathComponent(variant.checkpoint))
        let dit = try AuKDiT(weights: weights, frequencies: fusion.tensor("inv_freq").asArray(Float.self))
        let velocity = try dit.velocity(latent: zeros([1, 50, 64]), text: zeros([1, 16, 2048]),
                                        time: 0.3, reference: reference, guidance: variant == .flash ? 0 : 2)
        XCTAssertEqual(velocity.shape, [1, 50, 64])
        XCTAssertTrue(all(isFinite(velocity)).item(Bool.self))
    }

    func testTrainedCheckpointsMatchUpstream() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let outputPath = environment["AUK_QUALIFICATION_ROOT"],
              let modelPath = environment["AUK_MODEL_ROOT"],
              let thinkerPath = environment["AUK_THINKER_ROOT"] else {
            throw XCTSkip("Set AUK_QUALIFICATION_ROOT, AUK_MODEL_ROOT, AUK_THINKER_ROOT for trained parity")
        }
        let root = URL(fileURLWithPath: outputPath)
        let model = URL(fileURLWithPath: modelPath)
        let thinker = URL(fileURLWithPath: thinkerPath)
        let variant: AuKVariant = environment["AUK_VARIANT"] == "flash" ? .flash : .base
        func read(_ name: String) throws -> AuKTensorStore {
            AuKTensorStore(try loadArrays(url: root.appendingPathComponent(name + ".safetensors")))
        }
        var metrics = [String: Metric]()
        var native = [String: MLXArray]()
        func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 0.001) {
            eval(actual, expected)
            XCTAssertEqual(actual.shape, expected.shape, name)
            let difference = actual - expected
            let relative = sqrt(sum(difference * difference) / maximum(sum(expected * expected), MLXArray(Float(1e-20))))
                .item(Float.self)
            metrics[name] = Metric(shape: actual.shape, relativeL2: relative,
                                   maximumAbsolute: abs(difference).max().item(Float.self))
            native[name] = actual
            XCTAssertTrue(relative.isFinite, name)
            XCTAssertLessThan(relative, tolerance, name)
            print("AuK trained parity \(name): relative L2 \(relative)")
        }
        let inputs = try read("inputs"), conditioning = try read("conditioning")
        let diffusion = try read("diffusion"), waveforms = try read("waveforms")
        let reference = try read("reference").tensor("latent")
        let checkpoint = model.appendingPathComponent(variant.checkpoint)
        let config = try AuKThinkerConfiguration.load(from: thinker.appendingPathComponent("config.json"))
        do {
            let fusion = try AuKCheckpoint.diffusion(checkpoint).1
            let encoder = try AuKThinker(weights: AuKCheckpoint.thinker(thinker), configuration: config)
            for name in ["text", "audio"] {
                let encoded = try encoder.encode(tokens: inputs.tensor(name + "_ids").asArray(Int.self),
                    audio: name == "audio" ? inputs.tensor("audio_mel") : nil,
                    layerWeights: fusion.tensor("layer_weights"), layerScale: fusion.tensor("layer_scale"))
                compare(name + "_conditioning", encoded, try conditioning.tensor(name))
            }
        }
        Memory.clearCache()
        do {
            let vae = try AuKVAE(weights: AuKCheckpoint.vae(model.appendingPathComponent("vae.safetensors")))
            compare("reference_latent", try vae.encode(inputs.tensor("wave24")), reference)
            for name in ["text", "audio"] {
                compare(name + "_decoder", try vae.decode(diffusion.tensor(name + "_latent")), try waveforms.tensor(name))
            }
        }
        Memory.clearCache()
        do {
            let (weights, fusion) = try AuKCheckpoint.diffusion(checkpoint)
            let dit = try AuKDiT(weights: weights, frequencies: fusion.tensor("inv_freq").asArray(Float.self))
            let initial = try diffusion.tensor("initial")
            let schedule = try diffusion.tensor("schedule").asArray(Float.self)
            for name in ["text", "audio"] {
                let text = try conditioning.tensor(name)
                let ref = name == "audio" ? reference : nil
                let guidance: Float = variant == .flash ? 0 : 2
                compare(name + "_velocity", try dit.velocity(latent: initial, text: text, time: schedule[0],
                    reference: ref, guidance: guidance), try diffusion.tensor(name + "_velocity"))
                let nativeText = try XCTUnwrap(native[name + "_conditioning"])
                let nativeReference = name == "audio" ? try XCTUnwrap(native["reference_latent"]) : nil
                let latent = try AuKSampling.integrate(initial: initial, schedule: schedule) { state, time in
                    try dit.velocity(latent: state, text: nativeText, time: time,
                                     reference: nativeReference, guidance: guidance)
                }
                compare(name + "_trajectory", latent, try diffusion.tensor(name + "_latent"))
            }
        }
        Memory.clearCache()
        do {
            let vae = try AuKVAE(weights: AuKCheckpoint.vae(model.appendingPathComponent("vae.safetensors")))
            for name in ["text", "audio"] {
                let latent = try XCTUnwrap(native[name + "_trajectory"])
                compare(name + "_end_to_end", try vae.decode(latent), try waveforms.tensor(name), tolerance: 0.01)
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metrics).write(to: root.appendingPathComponent("native-parity.json"))
        try save(arrays: native, url: root.appendingPathComponent("native-parity.safetensors"))
    }
}
