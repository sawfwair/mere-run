import Foundation

public struct MiniMaxH3ImageResources: Sendable {
    public static let modelID = "image-minimax-h3"
    public static let repository = "MiniMaxAI/MiniMax-H3-Image"
    public static let revision = "492fd8c4eca666f1085f57e30738c1e0d86b3439"
    public static let patterns = ["LICENSE", "README.md", "model_index.json", "pdd_config.json",
        "pdd_fused_heads.safetensors", "transformer/*", "vae/*", "tokenizer/*", "processor/*", "text_encoder/*"]
    public let rootURL: URL
    public init(rootURL: URL) { self.rootURL = rootURL }

    func mlxFormat() throws -> MiniMaxH3ImageMLXFormat? {
        guard FileManager.default.fileExists(atPath: rootURL.appending(path: MiniMaxH3ImageMLXFormat.filename).path) else { return nil }
        let format = try decode(MiniMaxH3ImageMLXFormat.self, MiniMaxH3ImageMLXFormat.filename)
        try format.validate()
        return format
    }

    public func configuration() throws -> MiniMaxH3ImageConfiguration {
        let value = try decode(MiniMaxH3ImageConfiguration.self, "transformer/config.json")
        guard value.isReleasedArchitecture else { throw invalid("Unsupported image transformer configuration.") }
        let schedule = try decode(Schedule.self, "pdd_config.json")
        guard schedule.steps == 6, schedule.sigmas.count == 7,
              zip(schedule.sigmas, MiniMaxH3ImageSampling.sigmas).allSatisfy({ abs($0 - $1) < 1e-7 }) else {
            throw invalid("The checkpoint must contain the released six-step PDD schedule.")
        }
        return value
    }

    public func vaeConfiguration() throws -> MiniMaxH3ImageVAEConfiguration {
        let value = try decode(MiniMaxH3ImageVAEConfiguration.self, "vae/config.json")
        guard value.isReleasedArchitecture, value.latentMean.count == 24, value.latentStandardDeviation.count == 24,
              value.latentMean.allSatisfy(\.isFinite), value.latentStandardDeviation.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw invalid("The image VAE requires 24 finite latent means and positive standard deviations.")
        }
        return value
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var files = ["LICENSE", "transformer/config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
                     "pdd_config.json", "pdd_fused_heads.safetensors", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                     "text_encoder/config.json"]
        do {
            if try mlxFormat() != nil { files.append(MiniMaxH3ImageMLXFormat.cacheFilename) }
        } catch { return [rootURL.appending(path: MiniMaxH3ImageMLXFormat.filename)] }
        for component in ["transformer", "text_encoder"] {
            let name = "\(component)/model.safetensors.index.json"
            files.append(name)
            do { files += try index(component).shardFilenames.map { "\(component)/\($0)" } }
            catch { return [rootURL.appending(path: name)] }
        }
        let missing = files.map { rootURL.appending(path: $0) }.filter {
            var directory: ObjCBool = false
            return !fileManager.fileExists(atPath: $0.path, isDirectory: &directory) || directory.boolValue
        }
        if !missing.isEmpty { return missing }
        do { _ = try configuration(); _ = try vaeConfiguration() }
        catch { return [rootURL.appending(path: "transformer/config.json"), rootURL.appending(path: "vae/config.json")] }
        return []
    }

    func index(_ component: String) throws -> HFSafetensorsIndex {
        let value = try decode(HFSafetensorsIndex.self, "\(component)/model.safetensors.index.json")
        guard !value.weightMap.isEmpty, value.shardFilenames.allSatisfy({
            !$0.isEmpty && !$0.contains("/") && !$0.contains("\\") && $0 != "." && $0 != ".." && $0.hasSuffix(".safetensors")
        }) else { throw invalid("Invalid \(component) shard index.") }
        return value
    }

    private func decode<T: Decodable>(_ type: T.Type, _ path: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: rootURL.appending(path: path)))
    }

    private struct Schedule: Decodable {
        let steps: Int
        let sigmas: [Float]
        enum CodingKeys: String, CodingKey { case steps = "num_inference_steps", sigmas }
    }

    func invalid(_ message: String) -> ImageGenerationIssue { .init("h3_image_checkpoint_invalid", message) }
}
