import Foundation

public enum MeBotModelCatalog {
    public static let mebotId = "text-chat-mebot"
    public static let mebotDisplayName = "Me"
    public static let mebotDescription = "Personal chat model"
    public static let mebotSize = "~2 GB"
    public static let mebotArchiveKey = "models/mebot-instruct.tar.gz"
    public static let mebotArchiveSize: Int64 = 2_052_847_048
    public static var mebotArchiveURL: URL? {
        MereRunModelSourceConfiguration.publicArchiveURL(for: mebotArchiveKey)
    }

    /// Returns the local model path if downloaded, nil otherwise.
    /// Uses `resolveModelDir` so callers honor the active model-store overrides.
    public static func resolveModelPath() -> String? {
        let fm = FileManager.default
        let modelDir = MereRunModelPaths.resolveModelDir(mebotId) { root in
            fm.fileExists(atPath: root.appendingPathComponent("config.json").path)
        }
        if fm.fileExists(atPath: modelDir.appendingPathComponent("config.json").path) {
            return modelDir.path
        }
        return nil
    }
}
