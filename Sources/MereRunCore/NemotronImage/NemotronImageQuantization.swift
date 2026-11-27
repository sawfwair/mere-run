import Foundation
import MLX
import MLXNN

/// A standalone checkpoint with layer-local tensors in the Swift module layout.
public enum NemotronImageQuantization {
    public static let scheme = "nemotron-image-experts-affine-v1"

    static let mixedScheme = "nemotron-image-e6d8-affine-v1"

    static func expertBits(_ manifest: MereRunModelManifest?) throws -> Int? {
        guard let manifest else { return nil }
        if manifest.precision == .bf16, manifest.quantization == nil { return nil }
        let precisionBits: [MereRunModelManifest.Precision: Int] = [.int4: 4, .int6: 6, .int8: 8]
        guard let precision = manifest.precision, let bits = precisionBits[precision], let quant = manifest.quantization,
              quant.groupSize == 64, quant.bits == bits,
              quant.scheme == scheme || (quant.scheme == mixedScheme && bits == 6) else {
            throw NemotronImageResources.invalid("Expected a BF16 or group-64 affine expert checkpoint.")
        }
        return bits
    }

    static func filename(_ prefix: String) -> String {
        String(prefix.dropLast()) + ".safetensors"
    }

    static var files: [String] {
        (["backbone.embeddings.", "backbone.norm_f.", "lm_head."] + (0..<52).map { "backbone.layers.\($0)." }).map(filename)
    }

    static func validate(_ weights: [String: MLXArray], for module: Module) throws {
        let parameters = module.parameters().flattened()
        guard weights.count == parameters.count else {
            throw NemotronImageResources.invalid("Unexpected tensor count in packed module.")
        }
        for (key, parameter) in parameters {
            let packed = parameter.dtype == .uint32
            guard let value = weights[key], value.shape == parameter.shape, value.dtype == (packed ? .uint32 : .bfloat16) else {
                throw NemotronImageResources.invalid("Invalid packed checkpoint tensor \(key).")
            }
        }
    }

    static func pack(_ weights: [(String, MLXArray)], bits: Int, denseWeights: Set<String> = []) -> [String: MLXArray] {
        var output: [String: MLXArray] = [:]
        for (key, value) in weights {
            if (key.hasPrefix("mixer.experts.") && key.hasSuffix(".weight")) || denseWeights.contains(key) {
                let prefix = key.dropLast("weight".count)
                let bank = quantized(value, groupSize: 64, bits: denseWeights.contains(key) ? 8 : bits)
                output[key] = bank.wq
                output[prefix + "scales"] = bank.scales
                output[prefix + "biases"] = bank.biases
            } else {
                output[key] = value
            }
        }
        return output
    }

    static func denseWeightKeys(_ module: Module) -> Set<String> {
        if module is Linear { return ["weight"] }
        return Set(module.leafModules().flattened().compactMap { path, layer in
            layer is Linear ? path + ".weight" : nil
        })
    }

    public static func export(source: URL, destination: URL, bits requestedBits: Int?,
                              progress: (String) -> Void) throws -> [URL] {
        let fm = FileManager.default
        let community = NemotronImageCommunityPack.isPresent(at: source)
        let variant = try community ? NemotronImageCommunityPack.variant(at: source) : nil
        let bits = requestedBits ?? variant?.rawValue ?? 6
        guard [4, 6, 8].contains(bits), !fm.fileExists(atPath: destination.path) else {
            throw NemotronImageResources.invalid("Choose 4, 6 or 8 bits and a new output directory.")
        }
        let config = try NemotronImageResources.languageConfig(at: source)
        let decoder = community ? "vq" : NemotronImageResources.decoderDirectory
        try NemotronImageResources.DecoderConfig.validate(at: source.appending(path: decoder))
        var manifest: MereRunModelManifest
        let sourceModel: NemotronOmniCausalLM
        let index: HFSafetensorsIndex?
        if let variant {
            guard variant.rawValue == bits else {
                throw NemotronImageResources.invalid("Packed input requires --bits \(variant.rawValue).")
            }
            manifest = .template(for: .nemotronImage)
            let metadata = source.appending(path: ".cache/huggingface/download/config.json.metadata")
            let installed = try MereRunModelManifest.loadIfPresent(from: source)
            let revision = installed?.sources?.first { $0.repository == variant.repository }?.revision
                ?? (try? String(contentsOf: metadata, encoding: .utf8))?.split(separator: "\n").first.map(String.init)
            manifest.sources = (manifest.sources ?? []) + [.init(role: "packed-generator-and-decoder", repository: variant.repository,
                                      revision: revision ?? "unspecified", destinationPath: ".")]
            sourceModel = NemotronImageLoader.makeModel(config, bits: bits, mixed: variant == .mixed)
            index = nil
        } else {
            manifest = try MereRunModelManifest.loadRequired(from: source)
            guard manifest.engine == .nemotronImage, try expertBits(manifest) == nil else {
                throw NemotronImageResources.invalid("Expected the original Nemotron Image checkpoint.")
            }
            sourceModel = NemotronOmniCausalLM(config: config)
            index = try NemotronImageLoader.releasedIndex(at: source)
        }

        let parent = destination.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appending(path: ".nemotron-image-\(UUID().uuidString)")
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { if fm.fileExists(atPath: staging.path) { try? fm.removeItem(at: staging) } }
        var names: [String] = []
        try NemotronImageLoader.visitModules(sourceModel) { module, prefix in
            progress(prefix)
            let weights: [String: MLXArray]
            if let index {
                let dense = bits == 6 ? denseWeightKeys(module) : []
                weights = pack(try NemotronImageLoader.releasedWeights(at: source, config: config,
                    index: index, module: module, prefix: prefix), bits: bits, denseWeights: dense)
            } else {
                weights = try NemotronImageCommunityPack.weights(at: source, prefix: prefix, module: module)
                try validate(weights, for: module)
            }
            let name = filename(prefix)
            try MLX.save(arrays: weights, url: staging.appending(path: name))
            names.append(name)
        }
        let decoderPath = NemotronImageResources.decoderDirectory
        try fm.createDirectory(at: staging.appending(path: decoderPath), withIntermediateDirectories: false)
        var copies = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                      "unitron_extension_manifest.json"].map { ($0, $0) }
        for name in ["config.json", "model.safetensors"] {
            copies.append(("\(decoder)/\(name)", "\(decoderPath)/\(name)"))
        }
        for name in ["LICENSE", "NOTICE", "README.md"] where fm.fileExists(atPath: source.appending(path: name).path) {
            copies.append((name, name))
        }
        for (from, to) in copies {
            try fm.copyItem(at: source.appending(path: from), to: staging.appending(path: to))
            names.append(to)
        }
        manifest.id = destination.lastPathComponent
        switch bits {
        case 8: manifest.precision = .int8
        case 6: manifest.precision = .int6
        default: manifest.precision = .int4
        }
        manifest.quantization = .init(bits: bits, groupSize: 64, scheme: bits == 6 ? mixedScheme : scheme)
        manifest.createdAt = Date()
        try manifest.write(to: staging)
        try fm.moveItem(at: staging, to: destination)
        return (names + [MereRunModelManifest.filename]).map { destination.appending(path: $0) }
    }
}
