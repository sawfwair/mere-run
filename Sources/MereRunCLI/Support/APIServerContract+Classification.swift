import Foundation
import MereRunCore

struct GLiNERAPIRequest: Decodable, Sendable {
    let model: String
    let requests: [GLiNERClassificationRequest]
    let batch: Bool
    let long: Bool

    private enum CodingKeys: String, CodingKey { case model, requests, long }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        long = try container.decodeIfPresent(Bool.self, forKey: .long) ?? false
        if let array = try container.decodeIfPresent([GLiNERClassificationRequest].self, forKey: .requests) {
            requests = array
            batch = true
        } else {
            requests = [try GLiNERClassificationRequest(from: decoder)]
            batch = false
        }
    }

    func validate() throws {
        guard model == GLiNERCatalog.modelID else {
            throw APIRequestValidationError.invalidField("model", "select the installed GLiNER2.5 Decide model")
        }
        do {
            guard !requests.isEmpty else {
                throw GLiNERClassificationError.invalidRequest("Provide at least one request.")
            }
            try requests.forEach { try $0.validate() }
        } catch {
            throw APIRequestValidationError.invalidField("request", error.localizedDescription)
        }
    }
}

struct GLiNERExtractionAPIRequest: Decodable, Sendable {
    let model: String
    let requests: [GLiNERExtractionRequest]
    let batch: Bool
    let long: Bool

    private enum CodingKeys: String, CodingKey { case model, requests, long }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        long = try container.decodeIfPresent(Bool.self, forKey: .long) ?? false
        if let array = try container.decodeIfPresent([GLiNERExtractionRequest].self, forKey: .requests) {
            requests = array
            batch = true
        } else {
            requests = [try GLiNERExtractionRequest(from: decoder)]
            batch = false
        }
    }

    func validate() throws {
        guard model == GLiNERCatalog.modelID else {
            throw APIRequestValidationError.invalidField("model", "select the installed GLiNER2.5 Decide model")
        }
        do {
            guard !requests.isEmpty else {
                throw GLiNERClassificationError.invalidRequest("Provide at least one request.")
            }
            try requests.forEach { try $0.validate() }
        } catch {
            throw APIRequestValidationError.invalidField("request", error.localizedDescription)
        }
    }
}
