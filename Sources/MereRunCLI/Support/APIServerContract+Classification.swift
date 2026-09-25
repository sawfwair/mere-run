import Foundation
import MereRunCore

struct GLiNERAPIRequest: Decodable, Sendable {
    let model: String
    let request: GLiNERClassificationRequest

    private enum CodingKeys: String, CodingKey { case model }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        request = try GLiNERClassificationRequest(from: decoder)
    }

    func validate() throws {
        guard model == GLiNERCatalog.modelID else {
            throw APIRequestValidationError.invalidField("model", "select the installed GLiNER2.5 Decide model")
        }
        do {
            try request.validate()
        } catch {
            throw APIRequestValidationError.invalidField("request", error.localizedDescription)
        }
    }
}
