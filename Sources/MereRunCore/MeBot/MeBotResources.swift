import Foundation

public enum MeBotModelCatalog {
    public static let mebotId = "text-chat-mebot"
    public static let mebotDisplayName = "Me"
    public static let mebotDescription = "Personal chat model"
    public static let mebotSize = "~2 GB"

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
