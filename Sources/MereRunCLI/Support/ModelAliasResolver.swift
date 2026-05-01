import Foundation
import MereRunCore

enum ModelAliasResolver {
    static func modelID(for rawValue: String) -> ModelResolver.ModelID? {
        let normalized = rawValue
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return ModelResolver.ModelID(rawValue: normalized)
    }
}
