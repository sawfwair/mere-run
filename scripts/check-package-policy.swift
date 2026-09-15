import Foundation

struct Manifest: Decodable {
    var targets: [Target]
    var products: [Product]
    struct Target: Decodable {
        var name: String
        var type: String
        var dependencies: [Dependency]
    }
    struct Product: Decodable {
        var name: String
        var targets: [String]
    }
    struct Dependency: Decodable {
        var name: String
        var external: Bool
        var key: String { external ? "product:\(name)" : name }
        enum CodingKeys: String, CodingKey { case byName, target, product }
        init(name: String, external: Bool = false) { self.name = name; self.external = external }
        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            let key: CodingKeys = values.contains(.product) ? .product : values.contains(.target) ? .target : .byName
            var tuple = try values.nestedUnkeyedContainer(forKey: key)
            let name = try tuple.decode(String.self)
            external = key == .product
            if external {
                let package = try tuple.decode(String.self)
                self.name = "\(package)/\(name)"
            } else {
                self.name = name
            }
        }
    }
}

struct Policy: Decodable {
    var schemaVersion: Int
    var variants: [String]
    var existingImplementationProducts: [String]
    var targets: [Target]
    var products: [Product]
    var reexports: [Reexport]
    enum Role: String, Decodable {
        case library = "public-library", entry = "executable-entry-point", operation = "internal-operation"
        case model = "model-runtime", compatibility = "compatibility-surface", support = "test-support"
        case test = "test-target", binary = "binary-adapter", system = "system-adapter"
    }
    enum Exposure: String, Decodable { case sdk = "supported-sdk", tool = "supported-tool", compatibility, implementation = "implementation-exposure" }
    struct Target: Decodable {
        var name: String
        var variants: [String]
        var role: Role
        var manifestType: String
        var owner: String
        var purpose: String
        var requiresPath: String?
        var requiresSwiftSources: [String]?
        var allowedDirect: [String]
        var allowedTransitive: [String]
    }
    struct Product: Decodable {
        var name: String
        var variants: [String]
        var exposure: Exposure
        var owner: String
        var purpose: String
        var targets: [String]
        var consumers: [String]
        var disposition: String
        var removalCondition: String
        var requiresSwiftSources: [String]?
    }
    struct Reexport: Decodable {
        var file: String
        var module: String
        var owner: String
        var purpose: String
        var consumers: [String]
        var migration: String
        var removalCondition: String
        var key: String { "\(file):\(module)" }
    }
}

func nonempty(_ value: String) -> Bool { !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
func containsSwift(_ directory: URL) -> Bool {
    guard let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return false }
    return files.contains { ($0 as? URL)?.pathExtension == "swift" }
}
func applicable(path: String?, sources: [String]?, root: URL) -> Bool {
    if let path, !FileManager.default.fileExists(atPath: root.appendingPathComponent(path).path) { return false }
    if let sources, !sources.allSatisfy({ containsSwift(root.appendingPathComponent($0)) }) { return false }
    return true
}
func duplicateNames(_ names: [String]) -> Set<String> {
    Set(Dictionary(grouping: names, by: { $0 }).filter { $0.value.count != 1 }.keys)
}
func closure(_ roots: [String], targets: [String: Manifest.Target]) -> Set<String> {
    var found: Set<String> = []
    var pending = roots
    while let name = pending.popLast() {
        guard found.insert(name).inserted, let target = targets[name] else { continue }
        for dependency in target.dependencies {
            if dependency.external { found.insert(dependency.key) }
            else { pending.append(dependency.key) }
        }
    }
    return found
}

func validate(_ manifest: Manifest, policy: Policy, variant: String, root: URL) -> [String] {
    var errors: [String] = []
    guard policy.schemaVersion == 1, policy.variants.contains(variant) else { return ["Unknown package policy version or variant: \(variant)"] }
    if !duplicateNames(policy.variants).isEmpty { errors.append("Duplicate manifest variant") }
    let records = policy.targets.filter { $0.variants.contains(variant) && applicable(path: $0.requiresPath, sources: $0.requiresSwiftSources, root: root) }
    let products = policy.products.filter { $0.variants.contains(variant) && applicable(path: nil, sources: $0.requiresSwiftSources, root: root) }
    for (label, names) in [("target classification", records.map(\.name)), ("product classification", products.map(\.name)),
                            ("manifest target", manifest.targets.map(\.name)), ("manifest product", manifest.products.map(\.name))] {
        for name in duplicateNames(names).sorted() { errors.append("Duplicate \(label): \(name)") }
    }
    guard errors.isEmpty else { return errors }
    let implementationProducts = Set(policy.products.filter { $0.exposure == .implementation }.map(\.name))
    if Set(policy.existingImplementationProducts) != implementationProducts || !duplicateNames(policy.existingImplementationProducts).isEmpty {
        errors.append("Stale or duplicate implementation exposure inventory")
    }
    let targets = Dictionary(uniqueKeysWithValues: manifest.targets.map { ($0.name, $0) })
    let classified = Dictionary(uniqueKeysWithValues: records.map { ($0.name, $0) })
    let targetNames = Set(targets.keys), recordNames = Set(classified.keys)
    for name in targetNames.subtracting(recordNames).sorted() { errors.append("Unclassified target: \(name)") }
    for name in recordNames.subtracting(targetNames).sorted() { errors.append("Stale target classification: \(name)") }
    let manifestProducts = Set(manifest.products.map(\.name)), recordProducts = Set(products.map(\.name))
    for name in manifestProducts.subtracting(recordProducts).sorted() { errors.append("Unclassified product: \(name)") }
    for name in recordProducts.subtracting(manifestProducts).sorted() { errors.append("Stale product classification: \(name)") }
    for record in policy.targets {
        if record.variants.isEmpty || !Set(record.variants).isSubset(of: Set(policy.variants)) || !duplicateNames(record.variants).isEmpty {
            errors.append("Invalid variants for target: \(record.name)")
        }
        if !nonempty(record.owner) || !nonempty(record.purpose) { errors.append("Missing target responsibility: \(record.name)") }
    }
    for record in records {
        guard let target = targets[record.name] else { continue }
        if target.type != record.manifestType { errors.append("Manifest type changed for \(record.name)") }
        let requiredType: String
        switch record.role {
        case .test: requiredType = "test"
        case .binary: requiredType = "binary"
        case .system: requiredType = "system"
        case .entry: requiredType = "executable"
        default: requiredType = "regular"
        }
        if target.type != requiredType { errors.append("Incorrect role for manifest type: \(record.name)") }
        let direct = Set(target.dependencies.map(\.key))
        for dependency in target.dependencies where !dependency.external && targets[dependency.name] == nil {
            errors.append("Unknown local dependency \(record.name) -> \(dependency.name)")
        }
        for name in direct.subtracting(record.allowedDirect).sorted() { errors.append("Forbidden direct dependency \(record.name) -> \(name)") }
        let transitive = closure([record.name], targets: targets).subtracting([record.name])
        for name in transitive.subtracting(record.allowedTransitive).sorted() { errors.append("Forbidden transitive dependency \(record.name) -> \(name)") }
        if record.role != .support && record.role != .test {
            for name in transitive where classified[name]?.role == .support || classified[name]?.role == .test {
                errors.append("Production target \(record.name) depends on test support: \(name)")
            }
        }
    }
    for record in policy.products {
        if record.variants.isEmpty || !Set(record.variants).isSubset(of: Set(policy.variants)) || !duplicateNames(record.variants).isEmpty {
            errors.append("Invalid variants for product: \(record.name)")
        }
        if !nonempty(record.owner) || !nonempty(record.purpose) || record.consumers.isEmpty || !record.consumers.allSatisfy(nonempty) {
            errors.append("Missing product responsibility or consumer: \(record.name)")
        }
        if record.exposure == .implementation || record.exposure == .compatibility {
            if !nonempty(record.disposition) || !nonempty(record.removalCondition) { errors.append("Missing product migration/removal condition: \(record.name)") }
        }
        if record.exposure == .implementation && !policy.existingImplementationProducts.contains(record.name) {
            errors.append("New implementation-only product: \(record.name)")
        }
    }
    for product in manifest.products {
        guard let record = products.first(where: { $0.name == product.name }) else { continue }
        if Set(record.targets) != Set(product.targets) { errors.append("Product roots changed: \(product.name)") }
        let dependencies = closure(product.targets, targets: targets)
        for name in product.targets where targets[name] == nil { errors.append("Unknown product root \(product.name) -> \(name)") }
        for name in dependencies where classified[name]?.role == .support || classified[name]?.role == .test {
            errors.append("Shipped product \(product.name) depends on test support: \(name)")
        }
    }
    return errors
}

func validateReexports(_ policy: Policy, root: URL) throws -> [String] {
    var actual: Set<String> = []
    let pattern = try NSRegularExpression(pattern: "@_exported\\s+import\\s+([A-Za-z0-9_]+)")
    for directory in ["Sources", "apps"] {
        guard let paths = FileManager.default.enumerator(at: root.appendingPathComponent(directory), includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { continue }
        for case let file as URL in paths where file.pathExtension == "swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            for match in pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
                guard let range = Range(match.range(at: 1), in: text) else { continue }
                let relative = String(file.path.dropFirst(root.path.count + 1))
                actual.insert("\(relative):\(text[range])")
            }
        }
    }
    var errors = duplicateNames(policy.reexports.map(\.key)).sorted().map { "Duplicate re-export classification: \($0)" }
    let registered = Set(policy.reexports.map(\.key))
    errors += actual.subtracting(registered).sorted().map { "Unclassified re-export: \($0)" }
    errors += registered.subtracting(actual).sorted().map { "Stale re-export classification: \($0)" }
    for entry in policy.reexports {
        if [entry.owner, entry.purpose, entry.migration, entry.removalCondition].contains(where: { !nonempty($0) }) || entry.consumers.isEmpty || !entry.consumers.allSatisfy(nonempty) {
            errors.append("Missing re-export responsibility/migration: \(entry.key)")
        }
    }
    return errors
}

func selfTest() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let target = Manifest.Target(name: "Runtime", type: "regular", dependencies: [])
    let base = Manifest(targets: [target], products: [.init(name: "SDK", targets: ["Runtime"])])
    let record = Policy.Target(name: "Runtime", variants: ["fixture"], role: .library, manifestType: "regular", owner: "runtime", purpose: "Test fixture", allowedDirect: [], allowedTransitive: [])
    let product = Policy.Product(name: "SDK", variants: ["fixture"], exposure: .sdk, owner: "runtime", purpose: "Test fixture SDK", targets: ["Runtime"], consumers: ["fixture consumer"], disposition: "Retain", removalCondition: "Not scheduled")
    let policy = Policy(schemaVersion: 1, variants: ["fixture"], existingImplementationProducts: [], targets: [record], products: [product], reexports: [])
    var cases: [(String, Manifest, Policy, String?)] = [("valid", base, policy, nil)]
    var manifest = base; manifest.targets.append(.init(name: "Unknown", type: "regular", dependencies: []))
    cases.append(("unknown target", manifest, policy, "Unclassified target"))
    var changed = policy; changed.targets.append(record)
    cases.append(("duplicate target", base, changed, "Duplicate target classification"))
    manifest = base; manifest.products.append(.init(name: "Unreviewed", targets: ["Runtime"]))
    cases.append(("unknown product", manifest, policy, "Unclassified product"))
    changed = policy; changed.products.append(product)
    cases.append(("duplicate product", base, changed, "Duplicate product classification"))
    manifest = base; manifest.targets = []
    cases.append(("stale target", manifest, policy, "Stale target classification"))
    manifest = base; manifest.products = []
    cases.append(("stale product", manifest, policy, "Stale product classification"))
    manifest = base
    manifest.targets[0].dependencies = [.init(name: "Bridge")]
    manifest.targets += [.init(name: "Bridge", type: "regular", dependencies: [.init(name: "Forbidden")]), .init(name: "Forbidden", type: "regular", dependencies: [])]
    changed = policy; changed.targets[0].allowedDirect = ["Bridge"]; changed.targets[0].allowedTransitive = ["Bridge"]
    var bridge = record; bridge.name = "Bridge"; bridge.allowedDirect = ["Forbidden"]; bridge.allowedTransitive = ["Forbidden"]
    var forbidden = record; forbidden.name = "Forbidden"
    changed.targets += [bridge, forbidden]
    cases.append(("forbidden transitive edge", manifest, changed, "Forbidden transitive dependency Runtime -> Forbidden"))
    manifest = base; manifest.targets[0].dependencies = [.init(name: "Fixtures")]
    manifest.targets.append(.init(name: "Fixtures", type: "regular", dependencies: []))
    changed = policy; changed.targets[0].allowedDirect = ["Fixtures"]; changed.targets[0].allowedTransitive = ["Fixtures"]
    var support = record; support.name = "Fixtures"; support.role = .support; changed.targets.append(support)
    cases.append(("test support leakage", manifest, changed, "Shipped product SDK depends on test support"))
    changed = policy; changed.products[0].exposure = .implementation
    cases.append(("new implementation exposure", base, changed, "New implementation-only product"))
    changed = policy; changed.targets[0].variants = ["other"]
    cases.append(("unsupported variant", base, changed, "Invalid variants"))
    for (name, input, rules, expected) in cases {
        let errors = validate(input, policy: rules, variant: "fixture", root: root)
        let passed = expected.map { expected in errors.contains { $0.contains(expected) } } ?? errors.isEmpty
        if !passed { throw NSError(domain: "PackagePolicy", code: 1, userInfo: [NSLocalizedDescriptionKey: "Fixture failed: \(name): \(errors)"]) }
    }
    print("Package policy regression fixtures passed (\(cases.count)).")
}

do {
    let arguments = Array(CommandLine.arguments.dropFirst())
    if arguments == ["--self-test"] { try selfTest() }
    else {
        guard arguments.count == 3 else { throw NSError(domain: "PackagePolicy", code: 64, userInfo: [NSLocalizedDescriptionKey: "Usage: check-package-policy.swift <manifest.json> <policy.json> <variant> | --self-test"]) }
        let decoder = JSONDecoder()
        let manifest = try decoder.decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[0])))
        let policy = try decoder.decode(Policy.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[1])))
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let errors = validate(manifest, policy: policy, variant: arguments[2], root: root) + (try validateReexports(policy, root: root))
        if !errors.isEmpty { throw NSError(domain: "PackagePolicy", code: 1, userInfo: [NSLocalizedDescriptionKey: errors.joined(separator: "\n")]) }
        print("Package policy passed for \(arguments[2]): \(manifest.targets.count) targets, \(manifest.products.count) products.")
    }
} catch {
    FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
    exit(1)
}
