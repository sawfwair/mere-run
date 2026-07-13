import Foundation
import MLX
import MLXNN

enum MMAudioBigVGANCheckpointLoader {
    static func load(url: URL, into model: MMAudioBigVGAN) throws {
        let archive = try PyTorchStateDictArchive(url: url)
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(archive.tensors.count)
        for descriptor in archive.tensors {
            let value = try archive.loadArray(for: descriptor, dtype: .float16)
            updates.append(contentsOf: MMAudioBigVGAN.mapCheckpointWeight(
                key: descriptor.name,
                value: value
            ))
        }
        try model.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
    }
}
