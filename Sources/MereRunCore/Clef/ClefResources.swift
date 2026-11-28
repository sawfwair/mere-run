import Foundation
import MereRunQwenModel

public enum ClefCatalog {
    public static let modelID = "text-decide-clef-4bit"
    public static let repository = "mlx-community/clef-4bit"
    public static let revision = "e0a23bd4406c15075b7473616429c46f3fd130a9"
    public static let flashModelID = "text-decide-clef-flash-4bit"
    public static let flashRepository = "mlx-community/clef-flash-4bit"
    public static let flashRevision = "6822f0f244ee9e19df76908ba3302f7fe40ceea6"
    public static let modelIDs = [modelID, flashModelID, ClefOmniCatalog.modelID]
    public static let files = ["config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
                               "joint_head_config.json", "joint_head.safetensors", "model.safetensors.index.json",
                               "model-*.safetensors", "LICENSE", "README.md"]
    public static let hubFallback = HubFallbackConfig(repoId: repository, revision: revision, patterns: files)

    public static let flashHubFallback = HubFallbackConfig(repoId: flashRepository, revision: flashRevision, patterns: files)

    public static func isOmni(root: URL) throws -> Bool {
        struct Identity: Decodable { let model_type: String }
        return try JSONDecoder().decode(Identity.self, from: Data(contentsOf: root.appending(path: "config.json"))).model_type == "qwen3_omni_moe"
    }

    public static func validate(root: URL, fileManager: FileManager = .default) -> [URL] {
        if (try? isOmni(root: root)) == true { return ClefOmniCatalog.validate(root: root, fileManager: fileManager) }
        var missing = Q35Resources(rootURL: root).validate(fileManager: fileManager)
        missing += ["tokenizer_config.json", "processor_config.json", "joint_head_config.json", "joint_head.safetensors"]
            .map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        let indexURL = root.appending(path: "model.safetensors.index.json")
        if fileManager.fileExists(atPath: indexURL.path) {
            do {
                let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: indexURL))
                missing += Set(index.weightMap.values).map { root.appending(path: $0) }
                    .filter { !fileManager.fileExists(atPath: $0.path) }
                if index.weightMap.isEmpty { missing.append(indexURL) }
            } catch { missing.append(indexURL) }
        }
        guard missing.isEmpty else { return missing }
        do { _ = try ClefResources(root: root).configuration() }
        catch { return [root.appending(path: "config.json"), root.appending(path: "joint_head_config.json"), root.appending(path: "processor_config.json")] }
        return []
    }
}

package struct ClefResources {
    package let root: URL

    package init(root: URL) { self.root = root }

    package func configuration() throws -> (backbone: Q35Config, head: ClefHeadConfiguration, processor: ClefProcessorConfiguration) {
        let decoder = JSONDecoder()
        let backbone = try decoder.decode(Q35Config.self, from: Data(contentsOf: root.appending(path: "config.json")))
        let head = try decoder.decode(ClefHeadConfiguration.self, from: Data(contentsOf: root.appending(path: "joint_head_config.json")))
        let processor = try decoder.decode(ClefProcessorConfiguration.self, from: Data(contentsOf: root.appending(path: "processor_config.json")))
        try head.validate(backboneHiddenSize: backbone.textConfig.hiddenSize)
        guard backbone.modelType == "qwen3_5", !backbone.textConfig.usesMoE,
              backbone.textConfig.vocabSize > 0, backbone.textConfig.numHiddenLayers > 0,
              let vision = backbone.visionConfig, vision.patchSize == 16, vision.temporalPatchSize == 2,
              vision.spatialMergeSize == 2, vision.deepstackVisualIndexes?.isEmpty ?? true,
              backbone.imageTokenId != nil, backbone.videoTokenId != nil,
              backbone.quantization?.mode == "affine", backbone.quantization?.bits == 4,
              backbone.quantization?.groupSize == 64 else {
            throw ClefError.invalidConfiguration("Clef requires its dense Qwen3.5 affine 4-bit/group-64 vision checkpoint.")
        }
        try processor.validate()
        return (backbone, head, processor)
    }
}

package struct ClefProcessorConfiguration: Decodable {
    package struct Size: Decodable { package let longest_edge: Int; package let shortest_edge: Int }
    package struct Processor: Decodable {
        package let patch_size: Int
        package let temporal_patch_size: Int
        package let merge_size: Int
        package let image_mean: [Double]
        package let image_std: [Double]
        package let rescale_factor: Double
        package let size: Size
        package let fps: Double?
        package let min_frames: Int?
        package let max_frames: Int?
    }
    package let image_processor: Processor
    package let video_processor: Processor

    package func validate() throws {
        for processor in [image_processor, video_processor] {
            guard processor.patch_size == 16, processor.temporal_patch_size == 2, processor.merge_size == 2,
                  processor.image_mean == [0.5, 0.5, 0.5], processor.image_std == [0.5, 0.5, 0.5],
                  abs(processor.rescale_factor - 1 / 255.0) < 1e-12,
                  processor.size.shortest_edge > 0, processor.size.longest_edge >= processor.size.shortest_edge,
                  processor.size.longest_edge <= 25_165_824 else {
                throw ClefError.invalidConfiguration("Unsupported Clef image/video processor settings.")
            }
        }
        guard video_processor.fps == 2, video_processor.min_frames == 4, video_processor.max_frames == 768 else {
            throw ClefError.invalidConfiguration("Unsupported Clef video frame sampling settings.")
        }
    }
}
