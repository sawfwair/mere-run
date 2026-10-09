import Foundation
import MLX

public struct QwenImage21Resources: Sendable {
    public static let modelID = "image-qwen-21"
    public static let repoID = "Qwen/Qwen-Image-2.1"
    public static let revision = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
    public static let referenceRevision = "8d3c30bfda9b511c00992f40cff4170a5502814d"
    public static let turboModelID = "image-qwen-21-turbo"
    public static let turboRepoID = "Qwen/Qwen-Image-2.1-Turbo"
    public static let turboRevision = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
    public static let samplingReferenceRevision = "da1d3829cf08d4f329b526d89e17cc035c049d8d"
    public static let patterns = ["LICENSE", "model_index.json", "processor/*", "text_encoder/*", "transformer/*", "vae/*", "scheduler/*"]
    public let rootURL: URL
    public init(rootURL: URL) { self.rootURL = rootURL }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var paths = ["model_index.json", "processor/tokenizer.json", "processor/tokenizer_config.json",
                     "text_encoder/config.json", "transformer/config.json",
                     "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "scheduler/scheduler_config.json"]
        // Transformers 5 saves the combined image/video processor in processor_config.json.
        let processor = "processor/processor_config.json"
        paths.append(fileManager.fileExists(atPath: rootURL.appending(path: processor).path)
            ? processor : "processor/preprocessor_config.json")
        for (directory, stem) in [("text_encoder", "model"), ("transformer", "diffusion_pytorch_model")] {
            let single = "\(directory)/\(stem).safetensors"
            let indexPath = single + ".index.json"
            if fileManager.fileExists(atPath: rootURL.appending(path: single).path) { paths.append(single) } else {
                paths.append(indexPath)
                if let data = try? Data(contentsOf: rootURL.appending(path: indexPath)),
                   let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) {
                    if index.weightMap.isEmpty || index.shardFilenames.contains(where: { URL(fileURLWithPath: $0).lastPathComponent != $0 }) {
                        return [rootURL.appending(path: indexPath)]
                    }
                    paths += index.shardFilenames.map { directory + "/" + $0 }
                } else if fileManager.fileExists(atPath: rootURL.appending(path: indexPath).path) {
                    return [rootURL.appending(path: indexPath)]
                }
            }
        }
        return paths.map { rootURL.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
    }

    func decode<T: Decodable>(_ path: String, as type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: rootURL.appending(path: path)))
    }

    func samplingSchedule(steps: Int, tokenCount: Int, shift: Float?, requiresSavedSchedule: Bool) throws -> [Float] {
        let pipeline = try decode("model_index.json", as: QwenImage21PipelineConfiguration.self)
        guard !requiresSavedSchedule || pipeline.sampleSigmas != nil else {
            throw QwenImage21Error.invalidConfiguration("Qwen Image 2.1 Turbo requires sample_sigmas in model_index.json.")
        }
        return try decode("scheduler/scheduler_config.json", as: QwenImage21Scheduler.self)
            .sigmas(steps: steps, tokenCount: tokenCount, shift: shift, sampleSigmas: pipeline.sampleSigmas)
    }

    func arrays(_ directory: String, stem: String) throws -> [String: MLXArray] {
        let single = rootURL.appending(path: directory + "/" + stem + ".safetensors")
        if FileManager.default.fileExists(atPath: single.path) {
            return try MLX.loadArrays(url: single).mapValues { $0.dtype == .uint32 ? $0 : $0.asType(.bfloat16) }
        }
        let index = try decode(directory + "/" + stem + ".safetensors.index.json", as: HFSafetensorsIndex.self)
        var result: [String: MLXArray] = [:]
        for shard in index.shardFilenames {
            guard URL(fileURLWithPath: shard).lastPathComponent == shard else {
                throw QwenImage21Error.invalidWeights("Shard filename must be a basename.")
            }
            let arrays = try MLX.loadArrays(url: rootURL.appending(path: directory + "/" + shard))
            for (key, value) in arrays {
                guard index.weightMap[key] == shard, result[key] == nil else {
                    throw QwenImage21Error.invalidWeights("Shard ownership mismatch for \(key).")
                }
                result[key] = value.dtype == .uint32 ? value : value.asType(.bfloat16)
            }
        }
        guard Set(result.keys) == Set(index.weightMap.keys) else {
            throw QwenImage21Error.invalidWeights("Indexed checkpoint is missing tensors.")
        }
        return result
    }

    func quantization(_ directory: String) throws -> QwenImage21Quantization? {
        try decode(directory + "/config.json", as: ComponentQuantization.self).quantization
    }

    private struct ComponentQuantization: Decodable {
        let quantization: QwenImage21Quantization?
    }
}

struct QwenImage21Scheduler: Decodable {
    let baseImageSeqLen: Int
    let maxImageSeqLen: Int
    let baseShift: Float
    let maxShift: Float
    let shiftTerminal: Float?
    let shift: Float?
    let useDynamicShifting: Bool
    let timeShiftType: String
    let invertSigmas: Bool
    let stochasticSampling: Bool
    let useBetaSigmas: Bool
    let useExponentialSigmas: Bool
    let useKarrasSigmas: Bool
    enum CodingKeys: String, CodingKey {
        case baseImageSeqLen = "base_image_seq_len", maxImageSeqLen = "max_image_seq_len"
        case baseShift = "base_shift", maxShift = "max_shift", shiftTerminal = "shift_terminal", shift
        case useDynamicShifting = "use_dynamic_shifting", timeShiftType = "time_shift_type"
        case invertSigmas = "invert_sigmas", stochasticSampling = "stochastic_sampling"
        case useBetaSigmas = "use_beta_sigmas", useExponentialSigmas = "use_exponential_sigmas", useKarrasSigmas = "use_karras_sigmas"
    }

    func sigmas(steps: Int, tokenCount: Int, shift: Float? = nil, sampleSigmas: [Float]? = nil) throws -> [Float] {
        guard steps > 1, tokenCount > 0, maxImageSeqLen > baseImageSeqLen,
              timeShiftType == "exponential", !invertSigmas, !stochasticSampling,
              !useBetaSigmas, !useExponentialSigmas, !useKarrasSigmas else {
            throw QwenImage21Error.invalidConfiguration("Unsupported flow scheduler or fewer than two steps.")
        }
        let input = sampleSigmas ?? (0..<steps).map { 1 - Float($0) / Float(steps) }
        guard input.count == steps, input.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }),
              zip(input, input.dropFirst()).allSatisfy({ $0 > $1 }) else {
            throw QwenImage21Error.invalidConfiguration("Saved sampling sigmas must be descending values in (0, 1] matching the step count.")
        }
        var shifted: [Float]
        if useDynamicShifting {
            let slope = (maxShift - baseShift) / Float(maxImageSeqLen - baseImageSeqLen)
            let mu = shift ?? (baseShift + slope * Float(tokenCount - baseImageSeqLen))
            guard mu.isFinite, exp(mu).isFinite, exp(mu) > 0 else {
                throw QwenImage21Error.invalidConfiguration("Non-finite scheduler shift.")
            }
            let exponent = exp(mu)
            shifted = input.map { exponent / (exponent + 1 / $0 - 1) }
        } else {
            guard shift == nil, let fixedShift = self.shift, fixedShift.isFinite, fixedShift > 0 else {
                throw QwenImage21Error.invalidConfiguration("Static flow schedules require a positive checkpoint shift and do not accept --sigma-shift.")
            }
            shifted = input.map { fixedShift * $0 / (1 + (fixedShift - 1) * $0) }
        }
        if let terminal = shiftTerminal {
            guard terminal.isFinite, terminal > 0, terminal < 1, shifted.last! < 1 else {
                throw QwenImage21Error.invalidConfiguration("Invalid terminal flow shift.")
            }
            let scale = (1 - shifted.last!) / (1 - terminal)
            shifted = shifted.map { 1 - (1 - $0) / scale }
        }
        return shifted + [0]
    }
}

struct QwenImage21PipelineConfiguration: Decodable {
    let sampleSigmas: [Float]?
    enum CodingKeys: String, CodingKey { case sampleSigmas = "sample_sigmas" }
}
