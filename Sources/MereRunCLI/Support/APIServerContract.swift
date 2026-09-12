import Foundation

struct APIHealthStatus: Codable, Equatable, Sendable {
    let status: String
}

enum APIRequestValidationError: LocalizedError, Equatable {
    case invalidPayload
    case invalidField(String, String)

    var errorDescription: String? {
        switch self {
        case .invalidPayload:
            return "Invalid request payload."
        case .invalidField(let field, let reason):
            return "Invalid '\(field)': \(reason)."
        }
    }
}

enum APIServerContract {
    static func healthStatus() -> APIHealthStatus {
        APIHealthStatus(status: "ok")
    }

    static func acceptsJSONContentType(_ rawValue: String?) -> Bool {
        guard let mediaType = rawValue?
            .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
            .first?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
            !mediaType.isEmpty
        else {
            return false
        }
        return mediaType == "application/json" || mediaType.hasSuffix("+json")
    }

    static func multipartBoundary(from rawValue: String?) -> String? {
        guard let pieces = rawValue?.split(separator: ";", omittingEmptySubsequences: true),
              let mediaType = pieces.first?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              mediaType == "multipart/form-data" else {
            return nil
        }
        for piece in pieces.dropFirst() {
            let pair = piece.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2,
                  pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "boundary" else {
                continue
            }
            var boundary = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if boundary.hasPrefix("\""), boundary.hasSuffix("\""), boundary.count >= 2 {
                boundary.removeFirst()
                boundary.removeLast()
            }
            return boundary.isEmpty ? nil : String(boundary)
        }
        return nil
    }

    static func decodeJSONRequest<Request: Decodable>(
        _ type: Request.Type,
        from data: Data
    ) throws -> Request {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw APIRequestValidationError.invalidPayload
        }
    }
}
