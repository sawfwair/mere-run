import Foundation

extension APIServerContract {
    static func optionalPositiveIntField(_ rawValue: String?, field: String) throws -> Int? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Int(rawValue), value > 0 else {
            throw APIRequestValidationError.invalidField(field, "must be greater than zero")
        }
        return value
    }

    static func normalizedModelID(_ rawValue: String?, defaultID: String) -> String {
        normalizedOptional(rawValue) ?? defaultID
    }

    static func normalizedOptional(_ rawValue: String?) -> String? {
        guard let rawValue else { return nil }
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
