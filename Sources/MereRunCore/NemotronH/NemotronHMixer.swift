import MLX
import MLXNN

enum NemotronHLayerCache {
    case mamba(NemotronHMambaCache)
    case attention(Gemma4AttentionCache)

    var offset: Int {
        switch self {
        case .mamba(let cache): cache.offset
        case .attention(let cache): cache.offset
        }
    }

    func fork() -> NemotronHLayerCache {
        switch self {
        case .mamba(let cache): .mamba(cache.fork())
        case .attention(let cache): .attention(cache.fork())
        }
    }
}

class NemotronHMixer: Module {
    func callAsFunction(_ x: MLXArray, cache: NemotronHLayerCache?) -> MLXArray {
        preconditionFailure("Nemotron-H mixer is abstract")
    }
}
