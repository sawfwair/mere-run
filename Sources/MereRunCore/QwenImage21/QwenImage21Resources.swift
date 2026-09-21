import Foundation
import MLX

public struct QwenImage21Resources: Sendable {
    public static let modelID = "image-qwen-21"
    public static let repoID = "Qwen/Qwen-Image-2.1"
    public static let revision = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
    public static let referenceRevision = "8d3c30bfda9b511c00992f40cff4170a5502814d"
    public static let patterns = ["LICENSE", "model_index.json", "processor/*", "text_encoder/*", "transformer/*", "vae/*", "scheduler/*"]
    public let rootURL: URL
    public init(rootURL: URL) { self.rootURL = rootURL }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var paths = ["model_index.json", "processor/tokenizer.json", "processor/tokenizer_config.json",
                     "processor/preprocessor_config.json", "text_encoder/config.json", "transformer/config.json",
                     "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "scheduler/scheduler_config.json"]
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

    func arrays(_ directory: String, stem: String) throws -> [String: MLXArray] {
        let single = rootURL.appending(path: directory + "/" + stem + ".safetensors")
        if FileManager.default.fileExists(atPath: single.path) { return try MLX.loadArrays(url: single).mapValues { $0.asType(.bfloat16) } }
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
                result[key] = value.asType(.bfloat16)
            }
        }
        guard Set(result.keys) == Set(index.weightMap.keys) else {
            throw QwenImage21Error.invalidWeights("Indexed checkpoint is missing tensors.")
        }
        return result
    }
}

struct QwenImage21Scheduler: Decodable {
    let baseImageSeqLen: Int
    let maxImageSeqLen: Int
    let baseShift: Float
    let maxShift: Float
    let shiftTerminal: Float
    let useDynamicShifting: Bool
    let timeShiftType: String
    let invertSigmas: Bool
    let stochasticSampling: Bool
    let useBetaSigmas: Bool
    let useExponentialSigmas: Bool
    let useKarrasSigmas: Bool
    enum CodingKeys: String, CodingKey {
        case baseImageSeqLen = "base_image_seq_len", maxImageSeqLen = "max_image_seq_len"
        case baseShift = "base_shift", maxShift = "max_shift", shiftTerminal = "shift_terminal"
        case useDynamicShifting = "use_dynamic_shifting", timeShiftType = "time_shift_type"
        case invertSigmas = "invert_sigmas", stochasticSampling = "stochastic_sampling"
        case useBetaSigmas = "use_beta_sigmas", useExponentialSigmas = "use_exponential_sigmas", useKarrasSigmas = "use_karras_sigmas"
    }

    func sigmas(steps: Int, tokenCount: Int, shift: Float? = nil) throws -> [Float] {
        guard steps > 1, tokenCount > 0, maxImageSeqLen > baseImageSeqLen,
              useDynamicShifting, timeShiftType == "exponential", !invertSigmas, !stochasticSampling,
              !useBetaSigmas, !useExponentialSigmas, !useKarrasSigmas,
              shiftTerminal > 0, shiftTerminal < 1 else {
            throw QwenImage21Error.invalidConfiguration("Unsupported flow scheduler or fewer than two steps.")
        }
        let slope = (maxShift - baseShift) / Float(maxImageSeqLen - baseImageSeqLen)
        let mu = shift ?? (baseShift + slope * Float(tokenCount - baseImageSeqLen))
        guard mu.isFinite else { throw QwenImage21Error.invalidConfiguration("Non-finite scheduler shift.") }
        let exponent = exp(mu)
        let shifted = (0..<steps).map { index -> Float in
            let sigma = 1 - Float(index) / Float(steps)
            return exponent / (exponent + 1 / sigma - 1)
        }
        let scale = (1 - shifted.last!) / (1 - shiftTerminal)
        return shifted.map { 1 - (1 - $0) / scale } + [0]
    }
}
