#if canImport(CoreML) && os(macOS) && arch(arm64)
@preconcurrency import CoreML
import Foundation

/// Native Core ML execution of the pinned text, image, and audio export.
public final class EmbeddingGemma2 {
    public let config: BundleConfig
    public let tokenizer: GemmaTokenizer
    private let root: URL
    private let table: TokenEmbeddingTable
    private var positionTable: Data?
    private var models: [String: MLModel] = [:]
    private var compiled: [String: URL] = [:]

    public init(bundle: URL) throws {
        root = bundle
        config = try JSONDecoder().decode(BundleConfig.self, from: Data(contentsOf: bundle.appending(path: "config.json")))
        try config.validate()
        tokenizer = try GemmaTokenizer(contentsOf: bundle.appending(path: "tokenizer.json"))
        guard tokenizer.bosID == config.bos_token_id, tokenizer.eosID == config.eos_token_id,
              tokenizer.padID == config.pad_token_id, tokenizer.vocabSize == config.token_table.shape[0] else {
            throw EmbedderError(description: "Tokenizer does not match the Core ML bundle.")
        }
        table = try TokenEmbeddingTable(data: Data(contentsOf: bundle.appending(path: config.token_table.file), options: .alwaysMapped),
                                        rows: config.token_table.shape[0], columns: config.hidden_size, scale: config.embed_scale)
    }

    deinit {
        models.removeAll()
        for url in compiled.values { try? FileManager.default.removeItem(at: url) }
    }

    private func model(package: String, function: String?, compute: MLComputeUnits) async throws -> MLModel {
        let key = package + ":" + (function ?? "main")
        if let existing = models[key] { return existing }
        let url: URL
        if let cached = compiled[package] { url = cached }
        else {
            url = try await MLModel.compileModel(at: root.appending(path: package))
            compiled[package] = url
        }
        let settings = MLModelConfiguration()
        settings.computeUnits = compute
        settings.functionName = function
        let loaded = try await MLModel.load(contentsOf: url, configuration: settings)
        models[key] = loaded
        return loaded
    }

    public func embed(sequence: MultimodalSequence) async throws -> [Float] {
        let length = try config.textLength(for: sequence.ids.count)
        guard sequence.slots.count * 512 == sequence.softTokens.count,
              Set(sequence.slots).count == sequence.slots.count,
              sequence.slots.allSatisfy({ sequence.ids.indices.contains($0) }) else {
            throw EmbedderError(description: "Invalid Core ML embedding sequence.")
        }
        var values = try table.embeddings(ids: sequence.ids + Array(repeating: config.pad_token_id, count: length - sequence.ids.count))
        for (row, slot) in sequence.slots.enumerated() {
            values.replaceSubrange((slot * 512)..<((slot + 1) * 512), with: sequence.softTokens[(row * 512)..<((row + 1) * 512)])
        }
        let mask = (0..<length).map { Float16($0 < sequence.ids.count ? 1 : 0) }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: array(values, shape: [1, length, 512])),
            "attention_mask": MLFeatureValue(multiArray: array(mask, shape: [1, length]))
        ])
        let package = length <= 512 ? "EmbeddingGemma2Text.mlpackage" : config.extended!.text_package
        let text = try await model(package: package, function: "embed_\(length)", compute: length <= 512 ? .cpuAndNeuralEngine : .cpuAndGPU)
        let result = try await text.prediction(from: features)
        guard let output = result.featureValue(for: "embedding")?.multiArrayValue,
              output.shape.map(\.intValue) == [1, 768], output.dataType == .float32 else {
            throw EmbedderError(description: "Core ML returned an invalid embedding output.")
        }
        return (0..<768).map { output[[0, NSNumber(value: $0)]].floatValue }
    }

    public func imageSequence(url: URL, capacity: Int = 280) async throws -> MultimodalSequence {
        if positionTable == nil {
            positionTable = try Data(contentsOf: root.appending(path: config.vision.position_table.file), options: .alwaysMapped)
        }
        let inputs = try ImagePreprocessor.prepare(url: url, positionTable: positionTable!, tokens: capacity)
        let n = capacity * 9
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "patches": MLFeatureValue(multiArray: array(inputs.patches, shape: [1, n, 768])),
            "positions": MLFeatureValue(multiArray: array(inputs.positions.map(Float16.init), shape: [1, n, 2])),
            "position_embeddings": MLFeatureValue(multiArray: array(inputs.positionEmbeddings, shape: [1, n, 768])),
            "valid": MLFeatureValue(multiArray: array(inputs.valid, shape: [1, n])),
            "pool": MLFeatureValue(multiArray: array(inputs.pool, shape: [capacity, n]))
        ])
        let vision = try await model(package: config.vision.package, function: "vision_\(capacity)", compute: .cpuAndGPU)
        let output = try await vision.prediction(from: features)
        let soft = try softTokens(output: output, name: "image_tokens", capacity: capacity, count: inputs.tokenCount)
        return mediaSequence(soft: soft, start: config.vision.boi_token_id, end: config.vision.eoi_token_id)
    }

    public func audioSequence(url: URL) async throws -> MultimodalSequence {
        let decoded = try AudioDecoder.decode(url: url)
        guard decoded.channelCount > 0, let first = decoded.channels.first,
              decoded.channels.allSatisfy({ $0.count == first.count }) else { throw EmbedderError(description: "Invalid decoded audio channels.") }
        var mono = [Float](repeating: 0, count: first.count)
        for channel in decoded.channels { for i in mono.indices { mono[i] += channel[i] } }
        for i in mono.indices { mono[i] /= Float(decoded.channelCount) }
        let samples = AudioResampler.resampleTo16k(mono, rate: decoded.sampleRate)
        return try await audioSequence(samples: samples)
    }

    public func audioSequence(samples: [Float]) async throws -> MultimodalSequence {
        guard samples.count > 160, samples.count <= config.maximumAudioSeconds * 16000, samples.allSatisfy(\.isFinite) else {
            throw EmbedderError(description: "Each Core ML audio segment must contain finite 16 kHz samples and be longer than 10 ms and at most \(config.maximumAudioSeconds) seconds. Split longer recordings into ordered segments.")
        }
        if samples.count > 160000, let extended = config.extended {
            return try await extendedAudioSequence(samples: samples, settings: extended)
        }
        var waveform = [Float](repeating: 0, count: 160160)
        waveform.replaceSubrange(160..<(160 + samples.count), with: samples)
        let frames = CoreMLAudioGeometry.validFrames(samples: samples.count)
        let mask = (0..<1000).map { Float16($0 < frames ? 1 : 0) }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "waveform": MLFeatureValue(multiArray: floatArray(waveform, shape: [1, 160160])),
            "frame_mask": MLFeatureValue(multiArray: array(mask, shape: [1, 1000]))
        ])
        let audio = try await model(package: config.audio.package, function: nil, compute: .cpuAndGPU)
        let output = try await audio.prediction(from: features)
        let soft = try softTokens(output: output, name: "audio_tokens", capacity: 250, count: CoreMLAudioGeometry.tokenCount(samples: samples.count))
        return mediaSequence(soft: soft, start: config.audio.boa_token_id, end: config.audio.eoa_token_id)
    }

    private func extendedAudioSequence(samples: [Float], settings: BundleConfig.Extended) async throws -> MultimodalSequence {
        let inputs = AudioFrontend.logMel(samples)
        let frames = settings.audio_frames
        let tokens = AudioFrontend.softTokenCount(frames: inputs.frames)
        let features = inputs.features + [Float](repeating: 0, count: (frames - inputs.frames) * 128)
        let named: [(String, [Float], [Int])] = [
            ("features", features, [1, frames, 128]),
            ("frame_mask", (0..<frames).map { $0 < inputs.frames ? 1 : 0 }, [1, frames]),
            ("half_mask", (0..<(frames / 2)).map { $0 < (inputs.frames + 1) / 2 ? 1 : 0 }, [1, frames / 2]),
            ("token_mask", (0..<(frames / 4)).map { $0 < tokens ? 1 : 0 }, [1, frames / 4])
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: Dictionary(uniqueKeysWithValues: named.map {
            ($0.0, MLFeatureValue(multiArray: try floatArray($0.1, shape: $0.2)))
        }))
        let audio = try await model(package: settings.audio_package, function: nil, compute: .cpuAndGPU)
        let output = try await audio.prediction(from: provider)
        let soft = try softTokens(output: output, name: "audio_tokens", capacity: frames / 4, count: tokens, dtype: .float32)
        return mediaSequence(soft: soft, start: config.audio.boa_token_id, end: config.audio.eoa_token_id)
    }

    private func mediaSequence(soft: [Float16], start: Int32, end: Int32) -> MultimodalSequence {
        let count = soft.count / 512
        return MultimodalSequence(ids: [config.bos_token_id, start] + Array(repeating: config.pad_token_id, count: count)
            + [end, config.eos_token_id], softTokens: soft, slots: Array(2..<(2 + count)))
    }

    private func softTokens(output: any MLFeatureProvider, name: String, capacity: Int, count: Int,
                            dtype: MLMultiArrayDataType = .float16) throws -> [Float16] {
        guard let matrix = output.featureValue(for: name)?.multiArrayValue,
              matrix.shape.map(\.intValue) == [1, capacity, 512], matrix.dataType == dtype,
              (1...capacity).contains(count) else { throw EmbedderError(description: "Invalid Core ML \(name) output.") }
        return (0..<count).flatMap { row in
            (0..<512).map { column in Float16(matrix[[0, NSNumber(value: row), NSNumber(value: column)]].floatValue) }
        }
    }

    private func array(_ values: [Float16], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float16)
        values.withUnsafeBufferPointer { array.dataPointer.assumingMemoryBound(to: Float16.self).update(from: $0.baseAddress!, count: values.count) }
        return array
    }

    private func floatArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
        values.withUnsafeBufferPointer { array.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: values.count) }
        return array
    }
}
#endif
