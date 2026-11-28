import Foundation

public struct SheetSage2Resources: Sendable {
    public static let modelID = "music-sheetsage2"
    public static let repository = "m-a-p/SheetSage2"
    public static let revision = "b83c3d79a6d98bcffaf8f9ed112c90aa887887bc"
    public static let parentRepository = "m-a-p/MERT-v2-FullSong"
    public static let parentRevision = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
    public let rootURL: URL
    public init(rootURL: URL) { self.rootURL = rootURL }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        ["config.json", "model.safetensors", "LICENSE", "mert2/config.json", "mert2/model.safetensors", "mert2/LICENSE"]
            .map { rootURL.appendingPathComponent($0) }.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}
