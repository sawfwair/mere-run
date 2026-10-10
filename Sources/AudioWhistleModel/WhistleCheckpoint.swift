import Foundation
import MLX

package enum WhistleError: Error, LocalizedError {
    case invalid(String)

    package var errorDescription: String? {
        switch self {
        case .invalid(let message): return "Whistle: \(message)"
        }
    }
}

/// Geometry carried by the original safetensors checkpoint, not the Hub's summary config.
package struct WhistleConfig: Decodable, Equatable {
    let text: Int
    let dim: Int
    let kvHeads: Int
    let qkDim: Int
    let vDim: Int
    let encLayers: Int
    let decLayers: Int
    let convKernel: Int
    let taps: Int
    let lanes: Int
    let encLanes: Int
    let sites: [Int]
    let orders: [Int]
    let slots: Int
    let hashes: Int
    let stem: Int
    let stemStages: Int
    let rope: Float
    let mels: Int
    let rate: Int
    let fft: Int
    let win: Int
    let hop: Int
    let maxTokens: Int
    let pad: Int
    let eos: Int
    let bos: Int
    let dtype: String

    package static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: data)
        guard config.text == 8192, config.dim == 512, config.kvHeads == 2,
              config.qkDim == 48, config.vDim == 64, config.encLayers == 8,
              config.decLayers == 8, config.convKernel == 9, config.taps == 3,
              config.lanes == 4, config.encLanes == 4, config.sites == [3, 7],
              config.orders == [2, 3], config.slots == 18432, config.hashes == 2,
              config.stem == 128, config.stemStages == 3, config.rope == 100000,
              config.mels == 80, config.rate == 16000, config.fft == 512,
              config.win == 400, config.hop == 160, config.maxTokens == 320,
              config.pad == 0, config.eos == 1, config.bos == 2, config.dtype == "bfloat16" else {
            throw WhistleError.invalid("unsupported checkpoint geometry")
        }
        return config
    }
}

package final class WhistleWeights {
    let arrays: [String: MLXArray]
    let permutations: [MLXArray]
    let packed: [String: [WhistleCactusQuant]]

    package convenience init(arrays: [String: MLXArray]) throws {
        try self.init(arrays: arrays, packed: [:])
    }

    init(arrays: [String: MLXArray], packed: [String: [WhistleCactusQuant]]) throws {
        let layoutURL = Bundle.module.url(forResource: "layout", withExtension: "json")!
        let layout = try JSONDecoder().decode([String: [Int]].self, from: Data(contentsOf: layoutURL))
        guard Set(arrays.keys).union(packed.keys) == Set(layout.keys) else {
            throw WhistleError.invalid("checkpoint tensor names do not match the released layout")
        }
        for (name, shape) in layout where packed[name] == nil {
            guard arrays[name]?.shape == shape, arrays[name]?.dtype == .float32 else {
                throw WhistleError.invalid("invalid shape or dtype for \(name)")
            }
        }
        self.arrays = arrays
        self.packed = packed
        let permutationURL = Bundle.module.url(forResource: "permutations", withExtension: "json")!
        permutations = try JSONDecoder().decode([[Int32]].self, from: Data(contentsOf: permutationURL)).map { MLXArray($0) }
    }

    func callAsFunction(_ name: String, layer: Int? = nil) -> MLXArray {
        let value = arrays[name]!
        return layer.map { value[$0] } ?? value
    }

    func project(_ input: MLXArray, _ name: String, layer: Int? = nil) -> MLXArray {
        if let matrices = packed[name] { return matrices[layer ?? 0].project(input) }
        let value = self(name, layer: layer)
        return matmul(input, name == "embedding/embedding" ? value.T : value)
    }

    func gather(_ name: String, indices: MLXArray) -> MLXArray {
        if let matrices = packed[name] { return matrices[0].gather(indices) }
        let array = arrays[name]!
        return take(array.reshaped(-1, array.dim(-1)), indices, axis: 0)
    }

    package func evaluate() {
        eval(Array(arrays.values), permutations, packed.values.flatMap { matrices in
            matrices.flatMap { [$0.packed, $0.norms, $0.codebook] }
        })
    }

    package static func load(from url: URL) throws -> WhistleWeights {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        guard let raw = metadata["config"] else {
            throw WhistleError.invalid("safetensors metadata is missing config")
        }
        _ = try WhistleConfig.decode(Data(raw.utf8))
        return try WhistleWeights(arrays: arrays)
    }
}
