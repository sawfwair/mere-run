import Foundation

public enum KolibriResources {
    public static let referenceModelID = "text-chat-kolibri1-reference"
    public static let mixedModelID = "text-chat-kolibri1-mixed-2bit"
    public static let eightBitModelID = "text-chat-kolibri1-8bit"
    public static let defaultContextLength = 8_192
    public static let sourceRevision = "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"

    public static func handles(modelSpec: String) -> Bool {
        if [referenceModelID, mixedModelID, eightBitModelID].contains(modelSpec) { return true }
        return localModelRoot(for: modelSpec) != nil
    }

    public static func localModelRoot(for modelSpec: String) -> String? {
        let root = URL(fileURLWithPath: modelSpec).standardizedFileURL
        let url = root.appendingPathComponent("config.json")
        struct Envelope: Decodable { let model_type: String }
        guard let data = try? Data(contentsOf: url), let envelope = try? JSONDecoder().decode(Envelope.self, from: data) else {
            return nil
        }
        return envelope.model_type == "kolibri1" ? root.path : nil
    }
}
