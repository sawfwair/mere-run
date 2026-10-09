import Foundation
import XCTest
import MLX
import MLXNN
import AudioQwen3ASRModel
@testable import MereRunCore
@testable import MereRunQwenModel

final class ClefOmniTests: MereRunCoreTestCase {
    private var fixture: URL { Bundle.module.resourceURL!.appending(path: "Fixtures/ClefOmni") }

    private func arrays() throws -> [String: MLXArray] { try MLX.loadArrays(url: fixture.appending(path: "reference.safetensors")) }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 0.00003,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        guard actual.shape == expected.shape else { return }
        XCTAssertLessThanOrEqual(MLX.abs(actual - expected).max().item(Float.self), tolerance, file: file, line: line)
    }

    func testNativeThinkerMatchesIndependentCausalMoEAndInterleavedPositions() throws {
        let resources = ClefOmniResources(root: fixture)
        let (config, _) = try resources.configuration()
        let model = Qwen3OmniThinker(config: config.thinkerConfig.textConfig)
        try resources.loadText(model)
        let reference = try arrays()
        let ids = try XCTUnwrap(reference["ids"]), positions = try XCTUnwrap(reference["positions"])
        assertClose(try model(ids: ids, positions: positions), try XCTUnwrap(reference["hidden"]))
        let deepstack = try (0..<3).map { try XCTUnwrap(reference["deepstack.\($0)"]) }
        assertClose(try model(ids: ids, positions: positions, visualIndices: [2, 4], deepstack: deepstack),
                    try XCTUnwrap(reference["visual_hidden"]))
        // A later token must not influence earlier states.
        var changed = ids
        changed[0, 10] = MLXArray(Int32(99))
        let output = try model(ids: changed, positions: positions)
        assertClose(output[0..., 0..<10, 0...], try XCTUnwrap(reference["hidden"])[0..., 0..<10, 0...])
    }

    func testNativeVisionRetainsAllDeepstackFeatures() throws {
        let resources = ClefOmniResources(root: fixture)
        let (config, _) = try resources.configuration()
        let model = Qwen3OmniVision(config: config.thinkerConfig.visionConfig)
        try resources.loadVision(model)
        let reference = try arrays()
        let output = try model(patches: XCTUnwrap(reference["patches"]).reshaped(1, 8, 1536), temporal: 2, height: 2, width: 2)
        assertClose(output.embeddings, try XCTUnwrap(reference["vision"]))
        XCTAssertEqual(output.deepstack.count, 3)
        for index in output.deepstack.indices { assertClose(output.deepstack[index], try XCTUnwrap(reference["vision_deepstack.\(index)"])) }
    }

    func testNativeAudioMatchesChunkedOfficialEncoder() throws {
        let resources = ClefOmniResources(root: fixture)
        let (config, _) = try resources.configuration()
        let audio = config.thinkerConfig.audioConfig
        let model = Qwen3ASRAudioTower(config: Qwen3ASRAudioEncoderConfig(
            dModel: audio.dModel, numHiddenLayers: audio.encoderLayers, numAttentionHeads: audio.encoderAttentionHeads,
            ffnDim: audio.encoderFfnDim, maxSourcePositions: audio.maxSourcePositions,
            outputDim: audio.outputDim, downsampleHiddenSize: audio.downsampleHiddenSize))
        try resources.loadAudio(model)
        let reference = try arrays()
        let mel = try XCTUnwrap(reference["mel"])
        assertClose(model(mel.expandedDimensions(axis: 0), checkpointPositionArithmetic: true)[0], try XCTUnwrap(reference["audio"]))
        XCTAssertEqual(ClefOmniMedia.Audio.tokenCount(frames: mel.dim(1)), try XCTUnwrap(reference["audio"]).dim(0))
    }

    func testUnequalAudioBatchMatchesWhisperPaddingAndPartialHop() throws {
        let reference = try arrays()
        for index in 0..<2 {
            let samples = try XCTUnwrap(reference["waveform.\(index)"]).asArray(Float.self)
            let clip = ClefOmniMedia.Audio(samples: samples, paddedSampleCount: 20_000)
            XCTAssertEqual(clip.frames, index == 0 ? 101 : 125)
            assertClose(clip.mel(), try XCTUnwrap(reference["waveform_mel.\(index)"]), tolerance: 0.0001)
        }
        XCTAssertEqual(ClefOmniMedia.Audio(samples: Array(repeating: 0, count: 16_001)).frames, 100)
    }

    func testJointHeadUsesUntiedOutputEmbeddingRows() throws {
        let resources = ClefOmniResources(root: fixture)
        let (config, headConfig) = try resources.configuration()
        let model = Qwen3OmniThinker(config: config.thinkerConfig.textConfig)
        try resources.loadText(model)
        let head = try ClefJointHead(configuration: headConfig)
        try head.load(MLX.loadArrays(url: fixture.appending(path: "joint_head.safetensors")), dtype: .float32)
        let reference = try arrays(), ids = try XCTUnwrap(reference["ids"])
        let fields = [ClefHeadField(type: 1, questionSpan: 1..<3, optionSpans: [3..<5, 5..<7]),
                      ClefHeadField(type: 0, questionSpan: 7..<8, optionSpans: [8..<9, 9..<11])]
        let lexical = fields.map { $0.optionSpans.map { model.lexical(ids[0, $0]) } }
        let logits = try head(hidden: XCTUnwrap(reference["hidden"])[0], fields: fields, lexical: lexical)
        for index in logits.indices { assertClose(logits[index], try XCTUnwrap(reference["head_logits.\(index)"])) }
        XCTAssertGreaterThan(MLX.abs(model.lexical(ids) - model.embedding(ids)).max().item(Float.self), 0.1)
    }

    func testOmniRequestSupportsMixedLocalMediaAnd64000Tokens() throws {
        let data = Data(#"{"state":"Review","images":["a.png"],"audio":["call.wav"],"videos":["dashcam.mp4"],"questions":{"signal":{"type":"noul","instructions":false}}}"#.utf8)
        let request = try ClefDecisionRequest.decode(data, omni: true)
        XCTAssertEqual(request.maxTokens, 64_000)
        XCTAssertEqual(request.audio, ["call.wav"])
        XCTAssertEqual(request.videoFiles, ["dashcam.mp4"])
        XCTAssertEqual(try request.questions[0].instructions.rendered(), "false")
        XCTAssertThrowsError(try ClefDecisionRequest.decode(data))
        let excessive = Data(#"{"state":"x","questions":{"q":{"type":"noul"}},"max_tokens":64001}"#.utf8)
        XCTAssertThrowsError(try ClefDecisionRequest.decode(excessive, omni: true))
    }

    func testVideoSoundtrackTokensInterleaveByTimeWithoutDroppingEitherModality() {
        let visual = ClefOmniMedia.Visual(pixels: [], temporal: 2, height: 2, width: 4)
        let audio = ClefOmniMedia.Audio(samples: Array(repeating: 0, count: 16_000))
        let block = ClefOmniMedia.block(visual: visual, audio: audio, video: true)
        XCTAssertEqual(block.kinds.filter { $0 == 2 }.count, 4)
        XCTAssertEqual(block.kinds.filter { $0 == 3 }.count, 13)
        XCTAssertEqual(Array(block.kinds.prefix(3)), [2, 2, 3])
        XCTAssertEqual(Array(block.kinds.suffix(2)), [2, 2])
        XCTAssertEqual(block.bosCount, 2)
        XCTAssertTrue(block.text.hasPrefix("<|vision_start|><|audio_start|>"))
        XCTAssertTrue(block.text.hasSuffix("<|audio_end|><|vision_end|>\n"))
    }

    func testMixedMediaRotaryLayoutMatchesOfficialUpstream() throws {
        let (config, _) = try ClefOmniResources(root: fixture).configuration()
        let tc = config.thinkerConfig
        let markers = ["<|vision_start|>": tc.visionStartTokenId, "<|vision_end|>": tc.visionEndTokenId,
                       "<|audio_start|>": tc.audioStartTokenId, "<|audio_end|>": tc.audioEndTokenId,
                       "<|image_pad|>": tc.imageTokenId, "<|video_pad|>": tc.videoTokenId, "<|audio_pad|>": tc.audioTokenId]
        let tokenizer = ClefTokenizer(encode: { value in
            guard value.hasPrefix("<|vision_start|>") || value.hasPrefix("<|audio_start|>") else {
                return [value == "\n" ? 78 : 77]
            }
            var text = value, ids: [Int] = []
            while !text.isEmpty {
                if let marker = markers.first(where: { text.hasPrefix($0.key) }) {
                    ids.append(marker.value); text.removeFirst(marker.key.count)
                } else { ids.append(78); text.removeFirst() }
            }
            return ids
        })
        let clip = ClefOmniMedia.Audio(samples: Array(repeating: 0, count: 16_000))
        let media = ClefOmniMedia(blocks: [
            ClefOmniMedia.block(visual: .init(pixels: [], temporal: 1, height: 2, width: 4), audio: nil, video: false),
            ClefOmniMedia.block(visual: nil, audio: clip, video: false),
            ClefOmniMedia.block(visual: .init(pixels: [], temporal: 2, height: 2, width: 4), audio: clip, video: true),
        ])
        let reference = try arrays(), ids = try XCTUnwrap(reference["layout_ids"]).asArray(Int32.self).map(Int.init)
        XCTAssertEqual([77] + media.blocks.flatMap { tokenizer.encode($0.text) } + [77], ids)
        let request = try ClefDecisionRequest.decode(Data(#"{"state":"x","questions":{"q":{"type":"noul"}}}"#.utf8), omni: true)
        let plan = try tokenizer.sequence(request, modelID: ClefOmniCatalog.modelID).plan
        let sequence = ClefTokenSequence(ids: ids, fields: [], plan: plan)
        let layout = try ClefOmniLayout(media: media, tokenizer: tokenizer, sequence: sequence, config: config)
        assertClose(MLXArray(layout.positions.flatMap { $0 }, [3, 1, ids.count]),
                    try XCTUnwrap(reference["layout_positions"]), tolerance: 0)
    }

    func testPackedQ4ExpertsMatchReconstructedDenseThinker() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "omni-q4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var json = try ClefJSON.parse(Data(contentsOf: fixture.appending(path: "config.json")))
        // Use the typed JSON representation to create group-64-compatible synthetic geometry.
        func replacing(_ object: ClefJSON, _ key: String, _ value: ClefJSON) -> ClefJSON {
            .object((object.fields ?? []).filter { $0.key != key } + [.init(key: key, value: value)])
        }
        var thinker = try XCTUnwrap(json["thinker_config"])
        var text = try XCTUnwrap(thinker["text_config"])
        text = replacing(replacing(text, "hidden_size", .number("64")), "moe_intermediate_size", .number("64"))
        thinker = replacing(thinker, "text_config", text)
        thinker = replacing(thinker, "vision_config", replacing(try XCTUnwrap(thinker["vision_config"]), "out_hidden_size", .number("64")))
        thinker = replacing(thinker, "audio_config", replacing(try XCTUnwrap(thinker["audio_config"]), "output_dim", .number("64")))
        json = replacing(json, "thinker_config", thinker)
        let denseConfig = try Qwen3OmniConfiguration.decode(Data(try json.canonical().utf8))
        json = replacing(json, "quantization", try ClefJSON.parse(Data(#"{"bits":4,"group_size":64,"mode":"affine","scope":"thinker_moe_experts"}"#.utf8)))
        let packedConfig = try Qwen3OmniConfiguration.decode(Data(try json.canonical().utf8))
        let dense = Qwen3OmniThinker(config: denseConfig.thinkerConfig.textConfig)
        let packed = Qwen3OmniThinker(config: packedConfig.thinkerConfig.textConfig, quantization: packedConfig.quantization)
        var reconstructed: [String: MLXArray] = [:], disk: [String: MLXArray] = [:]
        for (key, value) in dense.parameters().flattened() {
            if key.contains(".mlp."), !key.contains(".mlp.gate.") {
                let (weight, scales, optionalBiases) = MLX.quantized(value, groupSize: 64, bits: 4)
                let biases = try XCTUnwrap(optionalBiases)
                reconstructed[key] = dequantized(weight, scales: scales, biases: biases, groupSize: 64, bits: 4)
                let components = key.split(separator: ".").map(String.init)
                for expert in 0..<4 {
                    let base = "thinker.model.layers.\(components[1]).mlp.experts.\(expert).\(components[3])"
                    disk[base + ".weight"] = weight[expert]
                    disk[base + ".scales"] = scales[expert]
                    disk[base + ".biases"] = biases[expert]
                }
            } else {
                reconstructed[key] = value
                disk[key == "lm_head.weight" ? "thinker." + key : "thinker.model." + key] = value
            }
        }
        try dense.update(parameters: ModuleParameters.unflattened(reconstructed), verify: [.all])
        try MLX.save(arrays: disk, url: root.appending(path: "model.safetensors"))
        // The disk index is deliberately written through the same public safetensor contract.
        let entries = disk.keys.sorted().map { "\"\($0)\":\"model.safetensors\"" }.joined(separator: ",")
        try Data("{\"weight_map\":{\(entries)}}".utf8).write(to: root.appending(path: "model.safetensors.index.json"))
        try ClefOmniResources(root: root).loadText(packed)
        let ids = MLXArray([Int32(1), 7, 11, 20], [1, 4])
        let positions = MLX.broadcast(MLXArray([Float(0), 1, 2, 3], [1, 1, 4]), to: [3, 1, 4])
        assertClose(try packed(ids: ids, positions: positions), try dense(ids: ids, positions: positions), tolerance: 0.0001)
        XCTAssertEqual(packed.parameters().flattened().filter { $0.0.contains(".mlp.gate_proj.weight") }.map { $0.1.dtype }, [.uint32, .uint32, .uint32])
    }

    func testOmniManagedCatalogPinsOriginalBF16Checkpoint() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ClefOmniCatalog.modelID))
        XCTAssertEqual(spec.upstreamRepoId, "Cloudflare/clef-omni")
        XCTAssertEqual(spec.upstreamRevision, "0db1cd2607d76a7bdb2a382f659e7b313079f84b")
        XCTAssertEqual(spec.validationKind, .clef)
        let manifest = try XCTUnwrap(MereRunModelManifest.template(for: .clefOmni))
        XCTAssertEqual(manifest.precision, .bf16)
    }
}
