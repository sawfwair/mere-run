import Foundation
import MereRunQwenModel

public enum ClairicCatalog {
    public static let modelID = "text-extract-clairic-bf16"
    public static let flashModelID = "text-extract-clairic-flash-bf16"
    public static let modelIDs = [flashModelID, modelID]

    static func repository(_ id: String) -> String {
        id == flashModelID ? "Cloudflare/clairic-flash" : "Cloudflare/clairic"
    }

    static func revision(_ id: String) -> String {
        id == flashModelID ? "36ad68b999560f56b57ef16a22180f8cb83c6d77" : "2c161be5cc71bacd6605b99a8c07547e58cfa434"
    }

    static func fallback(_ id: String) -> HubFallbackConfig {
        let shardPrefix = id == flashModelID ? flashShardPrefix : standardShardPrefix
        return HubFallbackConfig(repoId: repository(id), revision: revision(id),
                                 patterns: required + ["\(shardPrefix)*.safetensors", "LICENSE"])
    }

    public static func identifiesCheckpoint(at root: URL) -> Bool {
        FileManager.default.fileExists(atPath: root.appending(path: "clair_head_config.json").path)
    }

    public static func checkpointModelID(at root: URL) throws -> String {
        try ClairicConfiguration(root: root).isFlash ? flashModelID : modelID
    }

    private static let required = [
        "config.json", "clair_head_config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
        "model.safetensors.index.json",
    ]
    private static let flashShardPrefix = "model.safetensors-"
    private static let standardShardPrefix = "model-"

    static func validate(root: URL, fileManager: FileManager) -> [URL] {
        let missing = required.map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        if !missing.isEmpty { return missing }
        let indexURL = root.appending(path: "model.safetensors.index.json")
        guard let files = try? shardFilenames(at: indexURL) else { return [indexURL] }
        let missingShards = files.map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        if !missingShards.isEmpty { return missingShards }
        guard (try? ClairicConfiguration(root: root)) != nil else {
            return required.prefix(3).map { root.appending(path: $0) }
        }
        return []
    }

    static func shardFilenames(at url: URL) throws -> [String] {
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: url))
        let files = Set(index.weightMap.values)
        guard index.weightMap["lm_head.weight"] != nil, !files.isEmpty, files.allSatisfy({
            ($0.hasPrefix(flashShardPrefix) || $0.hasPrefix(standardShardPrefix))
                && $0.hasSuffix(".safetensors") && !$0.contains("/") && !$0.contains("\\")
        }) else {
            throw ClairicError.invalid("Clairic checkpoint index does not describe the released weight shards.")
        }
        return files.sorted()
    }
}

struct ClairicConfiguration {
    struct Processor: Decodable {
        struct Size: Decodable { let longest_edge: Int; let shortest_edge: Int }
        let size: Size
        let patch_size: Int
        let temporal_patch_size: Int
        let merge_size: Int
        let image_mean: [Double]
        let image_std: [Double]
    }

    private struct HeadIdentity: Decodable {
        let hidden_size: Int
        let readout: String
        let answer_slots: Bool
    }

    let backbone: Q35Config
    let processor: Processor
    let isFlash: Bool

    init(root: URL) throws {
        let decoder = JSONDecoder()
        backbone = try decoder.decode(Q35Config.self, from: Data(contentsOf: root.appending(path: "config.json")))
        processor = try decoder.decode(Processor.self, from: Data(contentsOf: root.appending(path: "preprocessor_config.json")))
        let head = try decoder.decode(HeadIdentity.self, from: Data(contentsOf: root.appending(path: "clair_head_config.json")))
        guard backbone.modelType == "qwen3_5", backbone.quantization == nil, !backbone.textConfig.usesMoE,
              [4_096, 5_120].contains(head.hidden_size), head.hidden_size == backbone.textConfig.hiddenSize,
              head.readout == "lexical", head.answer_slots, let vision = backbone.visionConfig,
              vision.patchSize == 16, vision.spatialMergeSize == 2, vision.temporalPatchSize == 2,
              backbone.imageTokenId != nil, backbone.visionStartTokenId != nil, backbone.visionEndTokenId != nil,
              processor.patch_size == 16, processor.temporal_patch_size == 2, processor.merge_size == 2,
              processor.image_mean == [0.5, 0.5, 0.5], processor.image_std == [0.5, 0.5, 0.5],
              processor.size.shortest_edge > 0,
              (processor.size.shortest_edge...16_777_216).contains(processor.size.longest_edge) else {
            throw ClairicError.invalid("Clairic requires the BF16 Cloudflare release and its Qwen3.5 image processor.")
        }
        isFlash = head.hidden_size == 4_096
    }
}
