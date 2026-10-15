import AudioDrama3Model
import Foundation
import Crypto
import MLX
import MereRunTensor
import XCTest
@testable import MereRunCore

final class Drama3QuantizationTests: MereRunCoreTestCase {
    func testCodecRejectsUnverifiedCheckpointBeforeParsing() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("unverified checkpoint".utf8).write(to: root.appendingPathComponent("codec.pth"))
        XCTAssertThrowsError(try Drama3Resources.loadCodec(root: root)) { error in
            XCTAssertTrue(error.localizedDescription.contains("pinned DRAMA-3 codec SHA-256"))
        }
    }

    private func digest(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    }

    func testNativeQ4ExportRoundTripPreservesSourceAndRecordsHashes() throws {
        let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = parent.appendingPathComponent("source"), output = parent.appendingPathComponent("q4")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parent) }
        let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Drama3RuntimeTests/Fixtures")
        var payload = try JSONDecoder().decode([String: OpenAIJSONValue].self,
            from: Data(contentsOf: fixtures.appendingPathComponent("tiny-config.json")))
        guard case .object(var text) = payload["text_config"], case .object(var audio) = payload["audio_decoder_config"] else {
            return XCTFail("invalid test configuration")
        }
        text["hidden_size"] = .number(128); text["vocab_size"] = .number(4_200)
        text["moe_intermediate_size"] = .number(128); text["shared_expert_intermediate_size"] = .number(128)
        audio["dim"] = .number(128)
        payload["text_config"] = .object(text); payload["audio_decoder_config"] = .object(audio)
        payload["eos_token_id"] = .number(42); payload["semantic_start_token_id"] = .number(100)
        payload["semantic_end_token_id"] = .number(4_195)
        let data = try JSONEncoder().encode(payload)
        try data.write(to: source.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Drama3Config.self, from: data)
        let model = try Drama3Model(config: config)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let published = try JSONDecoder().decode([String: OpenAIJSONValue].self,
            from: Data(contentsOf: fixtures.appendingPathComponent("nonexpert-shapes.json")))
        var arrays: [String: MLXArray] = [:]
        // Reuse upstream namespaces, with the full-attention layer 3 renamed to
        // the small fixture's sole layer 0. No released checkpoint is involved.
        for name in published.keys {
            if name.hasPrefix("model.language_model.layers.") && !name.hasPrefix("model.language_model.layers.3.") { continue }
            let original = name.replacingOccurrences(of: "model.language_model.layers.3.", with: "model.language_model.layers.0.")
            guard let mapped = Drama3Model.mappedKey(original), let target = parameters[mapped] else { continue }
            let shape = mapped.hasSuffix(".conv1d.weight") ? [target.dim(0), target.dim(2), target.dim(1)] : target.shape
            arrays[original] = MLXArray.zeros(shape, dtype: .bfloat16)
        }
        for expert in 0..<2 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let base = "model.language_model.layers.0.mlp.experts.\(expert).\(projection).weight"
                arrays[base] = MLXArray.full([128, 128], values: MLXArray(UInt8(0x38)))
                arrays[base + "_scale_inv"] = MLXArray([Float(3)]).reshaped(1, 1)
            }
        }
        let shard = source.appendingPathComponent("source.safetensors")
        try MLX.save(arrays: arrays, url: shard)
        // Replace U8 dtype tags with E4M3FN while retaining the exact raw bytes.
        let saved = try Data(contentsOf: shard)
        let oldLength = saved.withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self).littleEndian) }
        var header = try JSONDecoder().decode([String: OpenAIJSONValue].self, from: saved.subdata(in: 8..<(8 + oldLength)))
        for key in arrays.keys where key.contains(".experts.") && key.hasSuffix(".weight") {
            guard case .object(var value) = header[key] else { return XCTFail("missing test tensor") }
            value["dtype"] = .string("F8_E4M3"); header[key] = .object(value)
        }
        let encoded = try JSONEncoder().encode(header)
        var length = UInt64(encoded.count).littleEndian
        var packedFile = withUnsafeBytes(of: &length) { Data($0) }
        packedFile.append(encoded); packedFile.append(saved.dropFirst(8 + oldLength))
        try packedFile.write(to: shard)
        let index = HFSafetensorsIndex(metadata: nil,
            weightMap: Dictionary(uniqueKeysWithValues: arrays.keys.map { ($0, "source.safetensors") }))
        try JSONEncoder().encode(index).write(to: source.appendingPathComponent("model.safetensors.index.json"))
        for name in ["tokenizer.json", "tokenizer_config.json", "codec.pth", "LICENSE"] {
            // Packaging fixture: codec bytes are preserved, not evaluated.
            try Data("test-only packaging fixture".utf8).write(to: source.appendingPathComponent(name))
        }
        try model.validateLayout(Drama3Weights(root: source))
        let originalDigest = try digest(shard)
        let files = try Drama3QuantizedCheckpoint.export(source: source, destination: output)
        XCTAssertFalse(files.isEmpty)
        XCTAssertEqual(try digest(shard), originalDigest)
        XCTAssertEqual(try Data(contentsOf: source.appendingPathComponent("codec.pth")),
                       try Data(contentsOf: output.appendingPathComponent("codec.pth")))
        let converted = try Drama3Resources.configuration(root: output)
        XCTAssertEqual(converted.quantization?.bits, 4)
        let loaded = try Drama3Model(config: converted)
        try loaded.load(root: output)
        let native = try Drama3Weights(root: output)
        let base = "model.language_model.layers.0.mlp.experts.0.gate_proj"
        let restored = MLX.dequantized(try native.read(base + ".weight"), scales: try native.read(base + ".scales"),
            biases: try native.read(base + ".biases"), groupSize: 64, bits: 4)
        XCTAssertEqual(restored[0, 0].item(Float.self), 3, accuracy: 0.01)
        let receipt = try JSONDecoder().decode(Drama3QuantizedCheckpoint.Receipt.self,
            from: Data(contentsOf: output.appendingPathComponent("drama3-quantization.json")))
        XCTAssertTrue(receipt.sourceFiles.contains { $0.name == "source.safetensors" && $0.sha256 == originalDigest })
        for file in receipt.files {
            XCTAssertEqual(try digest(output.appendingPathComponent(file.name)), file.sha256)
        }
        XCTAssertThrowsError(try Drama3QuantizedCheckpoint.export(source: source, destination: output))
    }
}
