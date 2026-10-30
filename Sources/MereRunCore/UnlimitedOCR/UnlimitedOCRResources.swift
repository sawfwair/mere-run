import Foundation
import MereRunUnlimitedOCRModel

public enum UnlimitedOCRResources {
    public static let modelID = "vision-ocr-unlimited-2"
    public static let repository = "baidu/Unlimited-OCR-2"
    public static let revision = "c14071b549f4fcab10909f3c4acc6a2edb366947"
    public static let files = [
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    ]

    public static func missingFiles(root: URL, fileManager: FileManager = .default) -> [URL] {
        files.map { root.appendingPathComponent($0) }.filter { !fileManager.fileExists(atPath: $0.path) }
    }

    package static func configuration(root: URL) throws -> UnlimitedOCRConfiguration {
        try UnlimitedOCRConfiguration.load(Data(contentsOf: root.appendingPathComponent("config.json")))
    }
}
