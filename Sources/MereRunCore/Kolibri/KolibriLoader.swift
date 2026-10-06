import Foundation
import MLX
import MLXNN
import MereRunKolibriModel

enum KolibriLoader {
    struct Index: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    static func load(root: URL) throws -> KolibriCausalLM {
        let config = try JSONDecoder().decode(KolibriConfiguration.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let model = try KolibriCausalLM(config: config)
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
        guard Set(expected.keys) == Set(index.weightMap.keys) else {
            throw KolibriModelError.invalidWeights("Expected the complete stacked native layout; convert the upstream checkpoint first.")
        }
        for name in Set(index.weightMap.values).sorted() {
            try Task.checkCancellation()
            guard name == URL(fileURLWithPath: name).lastPathComponent, name.hasSuffix(".safetensors") else {
                throw KolibriModelError.invalidWeights("Invalid shard name.")
            }
            let arrays = try MLX.loadArrays(url: root.appendingPathComponent(name))
            let owned = Set(index.weightMap.filter { $0.value == name }.keys)
            guard owned == Set(arrays.keys) else { throw KolibriModelError.invalidWeights("Shard ownership mismatch in \(name).") }
            for (key, array) in arrays {
                guard let parameter = expected[key], parameter.shape == array.shape,
                      parameter.dtype == array.dtype else { throw KolibriModelError.invalidWeights("Shape or dtype mismatch for \(key).") }
            }
            try model.update(parameters: .unflattened(arrays), verify: .shapeMismatch)
            eval(Array(arrays.values))
        }
        return model
    }
}
