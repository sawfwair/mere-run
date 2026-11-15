import AudioSOVTranscribeModel
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
import Foundation
@testable import MereRunCore
import MereRunMLXTestSupport
import MLX
import MLXNN
import XCTest
@testable import AudioSTT

final class SOVPackedWeightsTests: MLXTestCase {
    private let configData = Data(#"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":3,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"max_position_embeddings":128,"rms_norm_eps":0.000001,"vocab_size":32,"enable_moe_block":true,"num_experts":4,"top_k_experts":2,"moe_intermediate_size":64}"#.utf8)

    private func model() throws -> Gemma4TextCausalLM {
        Gemma4TextCausalLM(config: try JSONDecoder().decode(Gemma4TextConfig.self, from: configData))
    }

    private func source(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try configData.write(to: root.appendingPathComponent("config.json"))
        let full = try model()
        full.prepareForFullPrecisionExpertWeights()
        let parameters = Dictionary(uniqueKeysWithValues: full.parameters().flattened())
        var arrays: [String: MLXArray] = [:]
        for (key, value) in parameters {
            if key.contains(".switch_glu.up_proj.") { continue }
            if key.contains(".switch_glu.gate_proj.weight") {
                let up = try XCTUnwrap(parameters[key.replacingOccurrences(of: "gate_proj", with: "up_proj")])
                let name = "model." + key.replacingOccurrences(of: "switch_glu.gate_proj.weight", with: "gate_up_proj")
                arrays[name] = concatenated([value, up], axis: 1).asType(.bfloat16)
            } else if key.contains(".switch_glu.down_proj.weight") {
                arrays["model." + key.replacingOccurrences(of: "switch_glu.down_proj.weight", with: "down_proj")] = value.asType(.bfloat16)
            } else {
                arrays["model." + key] = key.contains(".router.") ? value : value.asType(.bfloat16)
            }
        }
        try MLX.save(arrays: arrays, url: root.appendingPathComponent("model.safetensors"))
    }

    private func hashes(_ model: Module) -> [String: String] {
        // SOV disables per-layer input embeddings. Generic Gemma retains unused,
        // randomly initialized slots for that feature; they are absent from the
        // checkpoint and deliberately excluded by both loader completeness checks.
        Dictionary(uniqueKeysWithValues: model.parameters().flattened().filter {
            !$0.0.contains("per_layer") && !$0.0.contains(".rope.")
        }.map { name, value in
            (name, SOVPackedWeights.digest(value.asData(access: .noCopyIfContiguous).data))
        })
    }

    func testStreamingAndSavedWeightsMatchLegacyPackingForMixedPrecisions() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try source(at: root)
        let bits = [4, 6, 8]
        let legacy = try model()
        legacy.prepareForFullPrecisionExpertWeights()
        try SOVTranscribeWeights.load(root: root, into: legacy, target: true)
        try legacy.prepareExperimentalExperts(groupSize: 64, layerBits: bits)
        let expected = hashes(legacy)
        let destination = root.appendingPathComponent("packed")
        let streamed = try model()
        try streamed.preparePackedExpertWeights(groupSize: 64, layerBits: bits)
        let converted = try SOVPackedWeights.prepare(root: root, destination: destination, model: streamed, groupSize: 64, layerBits: bits)
        XCTAssertFalse(converted.cacheHit)
        XCTAssertLessThan(converted.packedBytes, converted.sourceBytes)
        XCTAssertEqual(hashes(streamed), expected)
        let restored = try model()
        try restored.preparePackedExpertWeights(groupSize: 64, layerBits: bits)
        let reused = try SOVPackedWeights.prepare(root: root, destination: destination, model: restored, groupSize: 64, layerBits: bits)
        XCTAssertTrue(reused.cacheHit)
        XCTAssertEqual(reused.sourceFingerprint, converted.sourceFingerprint)
        XCTAssertEqual(hashes(restored), expected)
        let x = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        XCTAssertEqual(abs(legacy(x) - restored(x)).max().item(Float.self), 0)
        XCTAssertThrowsError(try SOVPackedWeights.prepare(root: root, destination: destination, model: restored, groupSize: 64, layerBits: [8, 8, 8]))
        try Data("changed".utf8).write(to: root.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try SOVPackedWeights.prepare(root: root, destination: destination, model: restored, groupSize: 64, layerBits: bits))
    }

    func testFailedConversionDoesNotPublishPartialPackAndChangedShardIsRejected() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try source(at: root)
        let destination = root.appendingPathComponent("packed")
        let target = try model()
        try target.preparePackedExpertWeights(groupSize: 64, layerBits: [8, 8, 8])
        let complete = try Data(contentsOf: root.appendingPathComponent("model.safetensors"))
        try MLX.save(arrays: ["model.language_model.embed_tokens.weight": MLXArray.zeros([32, 64])],
                     url: root.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try SOVPackedWeights.prepare(root: root, destination: destination, model: target, groupSize: 64, layerBits: [8, 8, 8]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasSuffix(".partial") })
        try complete.write(to: root.appendingPathComponent("model.safetensors"))
        _ = try SOVPackedWeights.prepare(root: root, destination: destination, model: target, groupSize: 64, layerBits: [8, 8, 8])
        try Data().write(to: destination.appendingPathComponent("target-00001.safetensors"))
        XCTAssertThrowsError(try SOVPackedWeights.prepare(root: root, destination: destination, model: target, groupSize: 64, layerBits: [8, 8, 8]))
        XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent(SOVPackedWeights.manifestName).path))
    }

    func testInstalledPackedQ8MatchesEveryLegacyTargetParameter() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let sourcePath = env["MERERUN_TEST_SOV_PATH"],
              let packedPath = env["MERERUN_TEST_SOV_PACKED_PATH"] else {
            throw XCTSkip("Requires explicit original SOV and saved Q8 checkpoint paths on the GPU.")
        }
        let root = URL(fileURLWithPath: sourcePath), destination = URL(fileURLWithPath: packedPath)
        let config = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        XCTAssertEqual(config.textConfig.hiddenSizePerLayerInput, 0)
        let expected: [String: String] = try autoreleasepool {
            let legacy = Gemma4TextCausalLM(config: config.textConfig)
            legacy.prepareForFullPrecisionExpertWeights()
            try SOVTranscribeWeights.load(root: root, into: legacy, target: true)
            try legacy.prepareExperimentalExperts(groupSize: 64, bits: 8, metalVerification: true)
            return hashes(legacy)
        }
        Memory.clearCache()
        let target = Gemma4TextCausalLM(config: config.textConfig)
        let bits = Array(repeating: 8, count: config.textConfig.numHiddenLayers)
        try target.preparePackedExpertWeights(groupSize: 64, layerBits: bits, metalVerification: true)
        let result = try SOVPackedWeights.prepare(root: root, destination: destination, model: target, groupSize: 64, layerBits: bits)
        XCTAssertTrue(result.cacheHit)
        let actual = hashes(target)
        let mismatches = Set(actual.keys).union(expected.keys).filter { actual[$0] != expected[$0] }.sorted()
        XCTAssertEqual(mismatches, [])
        if let output = env["MERERUN_TEST_SOV_PACKED_RECEIPT"] {
            struct Receipt: Encodable {
                let sourceFingerprint: String
                let legacy: [String: String]
                let packed: [String: String]
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
            try encoder.encode(Receipt(sourceFingerprint: result.sourceFingerprint, legacy: expected, packed: actual))
                .write(to: URL(fileURLWithPath: output), options: .atomic)
        }
        if mismatches.isEmpty {
            print("SOV packed Q8 verified \(expected.count) checkpoint parameter byte digests; source \(result.sourceFingerprint)")
        }
    }

    func testPortableBundleLoadsWithoutSourceOrOriginalTimestamps() throws {
        let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: parent) }
        let root = parent.appendingPathComponent("source"), packed = parent.appendingPathComponent("cache")
        let bundle = parent.appendingPathComponent("download")
        try source(at: root)
        let target = try model(), bits = [8, 8, 8]
        try target.preparePackedExpertWeights(groupSize: 64, layerBits: bits)
        _ = try SOVPackedWeights.prepare(root: root, destination: packed, model: target, groupSize: 64, layerBits: bits)
        let expected = hashes(target)
        try FileManager.default.copyItem(at: packed, to: bundle)
        try FileManager.default.removeItem(at: bundle.appendingPathComponent("sov-packed.json"))
        try configData.write(to: bundle.appendingPathComponent("config.json"))
        try MLX.save(arrays: ["model.audio_tower.test": MLXArray.zeros([2, 2])], url: bundle.appendingPathComponent("model.safetensors"))
        func record(_ name: String) throws -> SOVTranscribeBundleManifest.FileRecord {
            let file = try SOVPackedWeights.FileRecord(root: bundle, name: name)
            return .init(name: name, bytes: file.bytes, headerSHA256: file.headerSHA256,
                         sha256: SOVPackedWeights.digest(try Data(contentsOf: bundle.appendingPathComponent(name))))
        }
        let manifest = try SOVTranscribeBundleManifest(formatVersion: 1, packerVersion: SOVPackedWeights.packerVersion,
            sourceRepository: SOVTranscribeResources.repository, sourceRevision: SOVTranscribeResources.revision,
            configSHA256: SOVPackedWeights.digest(configData), groupSize: 64, layerBits: bits,
            audio: record("model.safetensors"), shards: [record("target-00001.safetensors")])
        try JSONEncoder().encode(manifest).write(to: bundle.appendingPathComponent(SOVTranscribeBundleManifest.filename))
        try FileManager.default.removeItem(at: root)
        try FileManager.default.removeItem(at: packed)
        for file in [manifest.audio] + manifest.shards {
            try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 1)],
                                                  ofItemAtPath: bundle.appendingPathComponent(file.name).path)
        }
        let downloaded = try XCTUnwrap(SOVPortableBundle.read(root: bundle))
        try SOVPortableBundle.validateFiles(downloaded, root: bundle, layerCount: 3)
        let restored = try model()
        try restored.preparePackedExpertWeights(groupSize: 64, layerBits: bits)
        try SOVPortableBundle.load(downloaded, root: bundle, into: restored)
        XCTAssertEqual(hashes(restored), expected)
        let x = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        XCTAssertEqual(abs(target(x) - restored(x)).max().item(Float.self), 0)
        try FileManager.default.copyItem(at: bundle.appendingPathComponent(manifest.shards[0].name),
                                         to: bundle.appendingPathComponent("duplicate.safetensors"))
        let duplicate = try SOVTranscribeBundleManifest(formatVersion: manifest.formatVersion, packerVersion: manifest.packerVersion,
            sourceRepository: manifest.sourceRepository, sourceRevision: manifest.sourceRevision,
            configSHA256: manifest.configSHA256, groupSize: manifest.groupSize, layerBits: manifest.layerBits,
            audio: manifest.audio, shards: manifest.shards + [record("duplicate.safetensors")])
        try duplicate.validate()
        XCTAssertThrowsError(try SOVPortableBundle.load(duplicate, root: bundle, into: restored))
        XCTAssertThrowsError(try SOVPortableBundle.validateFiles(downloaded, root: bundle, layerCount: 2))
        try Data("changed".utf8).write(to: bundle.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try SOVPortableBundle.validateFiles(downloaded, root: bundle, layerCount: 3))
        try configData.write(to: bundle.appendingPathComponent("config.json"))
        try FileManager.default.removeItem(at: bundle.appendingPathComponent(downloaded.shards[0].name))
        XCTAssertThrowsError(try SOVPortableBundle.validateFiles(downloaded, root: bundle, layerCount: 3))
        XCTAssertTrue(SOVTranscribeResources.missingFiles(root: bundle, draft: false, fileManager: .default)
            .map(\.lastPathComponent).contains(downloaded.shards[0].name))
    }

    func testPortableDecoderSelectsQ8WithoutRequantizationAndRejectsOtherPrecisions() throws {
        XCTAssertEqual(try SOVPortableBundle.decoderPolicy(requested: .bf16, metalAvailable: true), .q8Metal)
        XCTAssertEqual(try SOVPortableBundle.decoderPolicy(requested: .bf16, metalAvailable: false), .q8)
        XCTAssertEqual(try SOVPortableBundle.decoderPolicy(requested: .q8, metalAvailable: true), .q8)
        for policy in SOVDecoderExperts.allCases where ![.bf16, .q8, .q8Metal].contains(policy) {
            XCTAssertThrowsError(try SOVPortableBundle.decoderPolicy(requested: policy, metalAvailable: true))
        }
        let hash = String(repeating: "a", count: 64)
        func manifest(_ name: String, version: Int = 1) -> SOVTranscribeBundleManifest {
            .init(formatVersion: version, packerVersion: SOVPackedWeights.packerVersion,
                sourceRepository: "source", sourceRevision: "revision", configSHA256: hash, groupSize: 64, layerBits: [8],
                audio: .init(name: "model.safetensors", bytes: 1, headerSHA256: hash, sha256: hash),
                shards: [.init(name: name, bytes: 1, headerSHA256: hash, sha256: hash)])
        }
        XCTAssertThrowsError(try manifest("../outside.safetensors").validate())
        XCTAssertThrowsError(try manifest("/outside.safetensors").validate())
        XCTAssertThrowsError(try manifest("model.safetensors").validate())
        XCTAssertThrowsError(try manifest("target.safetensors", version: 2).validate())
    }

    func testInstalledPortableBundleMatchesEveryTargetAndAudioParameter() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let sourcePath = env["MERERUN_TEST_SOV_PATH"],
              let packedPath = env["MERERUN_TEST_SOV_PACKED_PATH"], let bundlePath = env["MERERUN_TEST_SOV_BUNDLE_PATH"] else {
            throw XCTSkip("Requires explicit original, packed, and portable SOV paths on the GPU.")
        }
        let source = URL(fileURLWithPath: sourcePath), bundle = URL(fileURLWithPath: bundlePath)
        let config = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(contentsOf: bundle.appendingPathComponent("config.json")))
        let manifest = try XCTUnwrap(SOVPortableBundle.read(root: bundle))
        try SOVPortableBundle.validateFiles(manifest, root: bundle, layerCount: config.textConfig.numHiddenLayers)
        let expected: [String: String] = try autoreleasepool {
            let target = Gemma4TextCausalLM(config: config.textConfig)
            try target.preparePackedExpertWeights(groupSize: 64, layerBits: manifest.layerBits, metalVerification: true)
            _ = try SOVPackedWeights.prepare(root: source, destination: URL(fileURLWithPath: packedPath),
                model: target, groupSize: 64, layerBits: manifest.layerBits)
            return hashes(target)
        }
        Memory.clearCache()
        let actual: [String: String] = try autoreleasepool {
            let target = Gemma4TextCausalLM(config: config.textConfig)
            try target.preparePackedExpertWeights(groupSize: 64, layerBits: manifest.layerBits, metalVerification: true)
            try SOVPortableBundle.load(manifest, root: bundle, into: target)
            return hashes(target)
        }
        XCTAssertEqual(actual, expected)
        var audioCounts: [Int] = []
        for projector in [false, true] {
            let prefix = projector ? "model.embed_audio." : "model.audio_tower."
            func audioHashes(_ root: URL) throws -> [String: String] {
                try autoreleasepool {
                    let model: Module = projector ? SOVAudioProjector(config: config) : SOVAudioEncoder(config: config.audioConfig)
                    try SOVTranscribeWeights.load(root: root, into: model, sourcePrefix: prefix)
                    return hashes(model)
                }
            }
            let original = try audioHashes(source)
            let portable = try audioHashes(bundle)
            XCTAssertEqual(original, portable)
            audioCounts.append(original.count)
            Memory.clearCache()
        }
        if let output = env["MERERUN_TEST_SOV_BUNDLE_RECEIPT"] {
            struct Receipt: Encodable {
                let sourceRevision: String
                let target: [String: String]
                let encoderTensorCount: Int
                let projectorTensorCount: Int
                let targetEqual: Bool
            }
            try JSONEncoder().encode(Receipt(sourceRevision: manifest.sourceRevision, target: actual,
                encoderTensorCount: audioCounts[0], projectorTensorCount: audioCounts[1], targetEqual: actual == expected))
                .write(to: URL(fileURLWithPath: output), options: .atomic)
        }
    }
}
