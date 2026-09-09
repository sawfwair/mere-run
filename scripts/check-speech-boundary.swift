import Foundation

struct PackageManifest: Decodable {
    let targets: [Target]
    let products: [Product]

    struct Target: Decodable {
        let name: String
        let dependencies: [Dependency]
    }

    struct Product: Decodable {
        let name: String
        let targets: [String]
    }

    struct Dependency: Decodable {
        let name: String
        let isProduct: Bool

        enum CodingKeys: String, CodingKey {
            case byName, target, product
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let key: CodingKeys
            if container.contains(.target) {
                key = .target
            } else if container.contains(.byName) {
                key = .byName
            } else {
                key = .product
            }
            var value = try container.nestedUnkeyedContainer(forKey: key)
            name = try value.decode(String.self)
            isProduct = key == .product
        }
    }
}

struct DependencyClosure {
    var local: Set<String> = []
    var products: Set<String> = []
}

func dependencyClosure(_ roots: [String], targets: [String: PackageManifest.Target]) -> DependencyClosure {
    var result = DependencyClosure()
    var pending = roots
    while let name = pending.popLast() {
        guard result.local.insert(name).inserted, let target = targets[name] else { continue }
        for dependency in target.dependencies {
            if dependency.isProduct {
                result.products.insert(dependency.name)
            } else {
                pending.append(dependency.name)
            }
        }
    }
    return result
}

func checkBoundary(_ manifest: PackageManifest) -> [String] {
    let targets = Dictionary(uniqueKeysWithValues: manifest.targets.map { ($0.name, $0) })
    let modelTargets: Set<String> = [
        "AudioQwen3ASRModel", "AudioQwen3TTSModel", "AudioParakeetModel", "AudioSortformer"
    ]
    let supportTargets: Set<String> = ["MereRunKVCache", "MereRunModelKit"]
    let allowedProducts: Set<String> = ["MLX", "MLXFast", "MLXNN", "MLXRandom", "Crypto"]
    let checked = modelTargets.union(["MereRunKVCache", "MereRunMLXTestSupport", "SpeechRuntimeTests"])
    var failures: [String] = []
    for root in checked.sorted() {
        guard targets[root] != nil else {
            failures.append("Missing speech boundary target: \(root)")
            continue
        }
        let closure = dependencyClosure([root], targets: targets)
        let allowedLocal = root == "SpeechRuntimeTests"
            ? modelTargets.union(supportTargets).union([root, "MereRunMLXTestSupport"])
            : supportTargets.union([root])
        let unexpectedLocal = closure.local.subtracting(allowedLocal)
        let unexpectedProducts = closure.products.subtracting(allowedProducts)
        if !unexpectedLocal.isEmpty || !unexpectedProducts.isEmpty {
            failures.append(
                "\(root) has unexpected dependencies: "
                    + unexpectedLocal.union(unexpectedProducts).sorted().joined(separator: ", ")
            )
        }
    }
    for product in manifest.products {
        let closure = dependencyClosure(product.targets, targets: targets)
        if closure.local.contains("MereRunMLXTestSupport") {
            failures.append("Shipped product \(product.name) depends on MLX test support.")
        }
    }
    return failures
}

let manifestURL = URL(fileURLWithPath: CommandLine.arguments[1])
let manifest = try JSONDecoder().decode(PackageManifest.self, from: Data(contentsOf: manifestURL))
let failures = checkBoundary(manifest)
if failures.isEmpty {
    print("Speech model dependency boundaries passed.")
} else {
    FileHandle.standardError.write(Data((failures.joined(separator: "\n") + "\n").utf8))
    exit(1)
}
